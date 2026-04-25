from __future__ import annotations

from typing import Any

import httpx

from .config import get_app_credentials, get_user_access_token


class FeishuAPIError(RuntimeError):
    pass


class MissingFeishuConfigError(RuntimeError):
    pass


class FeishuClient:
    def __init__(
        self,
        base_url: str = "https://open.feishu.cn",
        user_access_token: str | None = None,
        tenant_access_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_access_token = user_access_token if user_access_token is not None else get_user_access_token()
        self._tenant_access_token = tenant_access_token

    def list_visible_chats(self, page_size: int = 100, page_token: str = "") -> dict[str, Any]:
        params: dict[str, Any] = {
            "user_id_type": "open_id",
            "page_size": min(max(page_size, 1), 100),
        }
        if page_token:
            params["page_token"] = page_token
        data = self.request(
            "GET",
            "/open-apis/im/v1/chats",
            access_token=self.get_tenant_access_token(),
            params=params,
        )
        body = data.get("data", data)
        return {
            "items": body.get("items", []),
            "has_more": bool(body.get("has_more", False)),
            "page_token": body.get("page_token", ""),
        }

    def list_chat_messages(
        self,
        chat_id: str,
        start_time: str = "",
        end_time: str = "",
        page_size: int = 50,
        page_token: str = "",
        sort: str = "asc",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "container_id_type": "chat",
            "container_id": chat_id,
            "sort_type": "ByCreateTimeAsc" if sort == "asc" else "ByCreateTimeDesc",
            "page_size": min(max(page_size, 1), 50),
            "card_msg_content_type": "raw_card_content",
        }
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if page_token:
            params["page_token"] = page_token

        if self.user_access_token:
            try:
                data = self.request("GET", "/open-apis/im/v1/messages", params=params)
            except FeishuAPIError:
                data = self.request(
                    "GET",
                    "/open-apis/im/v1/messages",
                    access_token=self.get_tenant_access_token(),
                    params=params,
                )
        else:
            data = self.request(
                "GET",
                "/open-apis/im/v1/messages",
                access_token=self.get_tenant_access_token(),
                params=params,
            )

        body = data.get("data", data)
        return {
            "items": body.get("items", []),
            "has_more": bool(body.get("has_more", False)),
            "page_token": body.get("page_token", ""),
        }

    def get_tenant_access_token(self) -> str:
        if self._tenant_access_token:
            return self._tenant_access_token
        app_id, app_secret = get_app_credentials()
        if not app_id or not app_secret:
            raise MissingFeishuConfigError(
                "Set FEISHU_APP_ID and FEISHU_APP_SECRET, or SOFREE_FEISHU_APP_ID and "
                "SOFREE_FEISHU_APP_SECRET, before collecting visible bot chats."
            )
        response = httpx.post(
            f"{self.base_url}/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": app_id, "app_secret": app_secret},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) not in (0, None):
            raise FeishuAPIError(f"tenant_access_token failed: {data}")
        token = data.get("tenant_access_token") or data.get("data", {}).get("tenant_access_token")
        if not token:
            raise FeishuAPIError(f"tenant_access_token response missing token: {data}")
        self._tenant_access_token = str(token)
        return self._tenant_access_token

    def request(
        self,
        method: str,
        path: str,
        access_token: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        token = access_token or self.user_access_token
        if not token:
            raise MissingFeishuConfigError("Missing Feishu user or tenant access token.")
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(base_url=self.base_url, timeout=30) as client:
            response = client.request(method, path, headers=headers, **kwargs)
        try:
            data = response.json()
        except ValueError:
            data = {"body": response.text}
        if response.is_error:
            raise FeishuAPIError(f"HTTP {response.status_code}: {data}")
        if isinstance(data, dict) and data.get("code", 0) not in (0, None):
            raise FeishuAPIError(str(data))
        return data
