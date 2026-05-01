from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import httpx

from .config import get_app_credentials, get_user_access_token
from .auth_token_store import TokenStore


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
        token_file: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_access_token = user_access_token
        self._tenant_access_token = tenant_access_token
        self.token_file = Path(token_file).expanduser() if token_file else None

    @classmethod
    def from_user_context(
        cls,
        *,
        token_file: str | Path | None = None,
        base_url: str = "https://open.feishu.cn",
        tenant_access_token: str | None = None,
        require_user_token: bool = False,
    ) -> FeishuClient:
        user_access_token = get_user_access_token(token_file=token_file)
        if require_user_token and not user_access_token:
            raise MissingFeishuConfigError(
                "Missing Feishu user access token. Run `sofree-knowledge auth login` first."
            )
        return cls(
            base_url=base_url,
            user_access_token=user_access_token,
            tenant_access_token=tenant_access_token,
            token_file=token_file,
        )

    def list_visible_chats(self, page_size: int = 100, page_token: str = "") -> dict[str, Any]:
        params: dict[str, Any] = {"user_id_type": "open_id", "page_size": min(max(page_size, 1), 100)}
        if page_token:
            params["page_token"] = page_token
        if self.user_access_token:
            try:
                data = self.request("GET", "/open-apis/im/v1/chats", params=params)
            except FeishuAPIError:
                data = self.request(
                    "GET",
                    "/open-apis/im/v1/chats",
                    access_token=self.get_tenant_access_token(),
                    params=params,
                )
        else:
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
            except (FeishuAPIError, MissingFeishuConfigError):
                # 用户token失败，且租户token配置不存在或也失败时，直接抛出给上层处理
                try:
                    data = self.request(
                        "GET",
                        "/open-apis/im/v1/messages",
                        access_token=self.get_tenant_access_token(),
                        params=params,
                    )
                except MissingFeishuConfigError:
                    # 没有配置租户凭证，直接抛出让上层跳过该群
                    raise
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

    def list_drive_files(self, page_size: int = 50, page_token: str = "") -> dict[str, Any]:
        params: dict[str, Any] = {
            "page_size": min(max(page_size, 1), 50),
            "order_by": "EditedTime",
            "direction": "DESC",
        }
        if page_token:
            params["page_token"] = page_token
        if self.user_access_token:
            data = self.request("GET", "/open-apis/drive/v1/files", params=params)
        else:
            data = self.request(
                "GET",
                "/open-apis/drive/v1/files",
                access_token=self.get_tenant_access_token(),
                params=params,
            )
        body = data.get("data", data)
        return {
            "items": body.get("files", body.get("items", [])),
            "has_more": bool(body.get("has_more", False)),
            "page_token": body.get("next_page_token", body.get("page_token", "")),
        }

    def get_doc_meta(self, doc_token: str, doc_type: str = "docx") -> dict[str, Any]:
        """Get document metadata by token. Supported types: docx, wiki, base, sheet, slides, file."""
        doc_type_map = {
            "docx": "docs",
            "wiki": "wiki",
            "base": "bitable",
            "sheet": "sheets",
            "slides": "slides",
            "file": "drive",
        }
        api_type = doc_type_map.get(doc_type, "docs")
        path = f"/open-apis/{api_type}/v1/{doc_type}s/{doc_token}"
        if doc_type == "file":
            path = f"/open-apis/drive/v1/files/{doc_token}"
        try:
            data = self.request("GET", path)
        except FeishuAPIError:
            # Fallback to tenant token if user token fails
            data = self.request(
                "GET",
                path,
                access_token=self.get_tenant_access_token(),
            )
        body = data.get("data", data)
        if doc_type == "file":
            return {
                "title": body.get("name", ""),
                "url": body.get("url", ""),
                "updated_at": body.get("modified_time", ""),
            }
        return {
            "title": body.get("title", ""),
            "url": body.get("url", ""),
            "updated_at": body.get("modified_time", ""),
        }

    def get_app_access_token(self) -> str:
        app_id, app_secret = get_app_credentials()
        if not app_id or not app_secret:
            raise MissingFeishuConfigError("FEISHU_APP_ID and FEISHU_APP_SECRET are required.")
        try:
            response = httpx.post(
                f"{self.base_url}/open-apis/auth/v3/app_access_token/internal",
                json={"app_id": app_id, "app_secret": app_secret},
                timeout=30,
            )
        except (httpx.HTTPError, OSError) as exc:
            raise FeishuAPIError(self._format_network_error(exc)) from exc
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) not in (0, None):
            raise FeishuAPIError(self._format_feishu_error(data, prefix="app_access_token failed"))
        token = data.get("app_access_token") or data.get("data", {}).get("app_access_token")
        if not token:
            raise FeishuAPIError(self._format_feishu_error(data, prefix="app_access_token response missing token"))
        return str(token)

    def create_lingo_entity(
        self,
        key: str,
        description: str,
        aliases: list[str] | None = None,
        provider: str = "sofree-knowledge-cli",
        outer_id: str | None = None,
    ) -> dict[str, Any]:
        display_status = {"allow_highlight": True, "allow_search": True}
        normalized_key = str(key or "").strip()
        normalized_desc = str(description or "").strip()
        if not normalized_key:
            raise ValueError("key is required for create_lingo_entity")
        if not normalized_desc:
            raise ValueError("description is required for create_lingo_entity")

        payload = {
            "main_keys": [{"key": normalized_key, "display_status": display_status}],
            "aliases": [
                {"key": alias, "display_status": display_status}
                for alias in (aliases or [])
                if str(alias).strip()
            ],
            "description": normalized_desc,
            "rich_text": f"<p><span>{html.escape(normalized_desc)}</span></p>",
        }
        try:
            data = self.request(
                "POST",
                "/open-apis/baike/v1/entities",
                access_token=self.get_app_access_token(),
                params={"user_id_type": "open_id"},
                json=payload,
            )
        except FeishuAPIError as exc:
            # Compatibility fallback for tenants that still accept outer_info only in some envs.
            message = str(exc)
            if "display_status" in message:
                raise FeishuAPIError(
                    f"{message}; hint=main_keys/aliases must include display_status."
                ) from exc
            raise
        body = data.get("data", data)
        return {
            "entity_id": str(body.get("id") or body.get("entity_id") or ""),
            "visibility_note": (
                "Create API success does not always mean the entry is immediately searchable. "
                "It may require admin review or repository visibility setup."
            ),
            "provider": provider,
            "outer_id": str(outer_id or normalized_key),
            "raw": body,
        }

    def delete_lingo_entity(self, entity_id: str) -> dict[str, Any]:
        data = self.request(
            "DELETE",
            f"/open-apis/baike/v1/entities/{entity_id}",
            access_token=self.get_app_access_token(),
        )
        body = data.get("data", data)
        return {
            "entity_id": str(entity_id),
            "raw": body,
        }

    def send_message(
        self,
        receive_id: str,
        msg_type: str,
        content: str | dict[str, Any],
        receive_id_type: str = "chat_id",
    ) -> dict[str, Any]:
        payload_content = content
        if isinstance(content, dict):
            payload_content = json.dumps(content, ensure_ascii=False)
        data = self.request(
            "POST",
            "/open-apis/im/v1/messages",
            access_token=self.get_tenant_access_token(),
            params={"receive_id_type": receive_id_type},
            json={
                "receive_id": receive_id,
                "msg_type": msg_type,
                "content": str(payload_content),
            },
        )
        body = data.get("data", data)
        return {
            "message_id": str(body.get("message_id", "")),
            "chat_id": str(body.get("chat_id", "")),
            "msg_type": str(body.get("msg_type", msg_type)),
            "raw": body,
        }

    def get_tenant_access_token(self) -> str:
        if self._tenant_access_token:
            return self._tenant_access_token
        app_id, app_secret = get_app_credentials()
        if not app_id or not app_secret:
            raise MissingFeishuConfigError("FEISHU_APP_ID and FEISHU_APP_SECRET are required.")
        try:
            response = httpx.post(
                f"{self.base_url}/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": app_id, "app_secret": app_secret},
                timeout=30,
            )
        except (httpx.HTTPError, OSError) as exc:
            raise FeishuAPIError(self._format_network_error(exc)) from exc
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) not in (0, None):
            raise FeishuAPIError(self._format_feishu_error(data, prefix="tenant_access_token failed"))
        token = data.get("tenant_access_token") or data.get("data", {}).get("tenant_access_token")
        if not token:
            raise FeishuAPIError(self._format_feishu_error(data, prefix="tenant_access_token response missing token"))
        self._tenant_access_token = str(token)
        return self._tenant_access_token

    def request(self, method: str, path: str, access_token: str | None = None, **kwargs: Any) -> dict[str, Any]:
        token = access_token or self.user_access_token
        if not token:
            raise MissingFeishuConfigError("Missing Feishu access token.")
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Authorization"] = f"Bearer {token}"
        try:
            with httpx.Client(base_url=self.base_url, timeout=30) as client:
                response = client.request(method, path, headers=headers, **kwargs)
                if self._should_retry_with_refreshed_user_token(response, access_token):
                    refreshed_token = self.refresh_user_access_token()
                    headers["Authorization"] = f"Bearer {refreshed_token}"
                    response = client.request(method, path, headers=headers, **kwargs)
        except (httpx.HTTPError, OSError) as exc:
            raise FeishuAPIError(self._format_network_error(exc)) from exc
        try:
            data = response.json()
        except ValueError:
            data = {"body": response.text}
        if response.is_error:
            raise FeishuAPIError(self._format_feishu_error(data, status_code=response.status_code))
        if isinstance(data, dict) and data.get("code", 0) not in (0, None):
            raise FeishuAPIError(self._format_feishu_error(data))
        return data

    def refresh_user_access_token(self) -> str:
        if not self.token_file:
            raise MissingFeishuConfigError("Missing token file for user token refresh. Re-run `sofree-knowledge auth login`.")
        store = TokenStore(token_file=self.token_file)
        token_data = store.load()
        refresh_token = str(token_data.get("refresh_token") or "").strip()
        if not refresh_token:
            raise MissingFeishuConfigError(
                "Missing Feishu refresh token. Re-run `sofree-knowledge auth login` to refresh authorization."
            )
        try:
            response = httpx.post(
                f"{self.base_url}/open-apis/authen/v1/refresh_access_token",
                json={"grant_type": "refresh_token", "refresh_token": refresh_token},
                headers={"Authorization": f"Bearer {self.get_app_access_token()}"},
                timeout=30,
            )
        except (httpx.HTTPError, OSError) as exc:
            raise FeishuAPIError(self._format_network_error(exc)) from exc
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) not in (0, None):
            raise FeishuAPIError(self._format_feishu_error(data, prefix="refresh_access_token failed"))
        refreshed = data.get("data", data)
        new_access_token = str(refreshed.get("access_token") or "").strip()
        if not new_access_token:
            raise FeishuAPIError(self._format_feishu_error(data, prefix="refresh_access_token missing access_token"))
        merged = dict(token_data)
        merged.update(refreshed)
        store.save(merged)
        self.user_access_token = new_access_token
        return new_access_token

    def _should_retry_with_refreshed_user_token(
        self,
        response: httpx.Response,
        access_token: str | None,
    ) -> bool:
        return (
            response.status_code == 401
            and access_token is None
            and bool(self.user_access_token)
            and self.token_file is not None
        )

    def _format_network_error(self, exc: Exception) -> str:
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 10013:
            return (
                "Network access denied (WinError 10013). "
                "Check firewall/proxy/sandbox permissions, then retry."
            )
        return f"Network request failed: {exc}"

    def _format_feishu_error(
        self,
        data: Any,
        *,
        status_code: int | None = None,
        prefix: str | None = None,
    ) -> str:
        if not isinstance(data, dict):
            base = f"Feishu API error: {data}"
            if status_code is not None:
                base = f"HTTP {status_code}: {base}"
            if prefix:
                return f"{prefix}: {base}"
            return base

        code = data.get("code")
        msg = str(data.get("msg", "")).strip()
        parts: list[str] = []
        if prefix:
            parts.append(prefix)
        if status_code is not None:
            parts.append(f"HTTP {status_code}")
        if code is not None:
            parts.append(f"code={code}")
        if msg:
            parts.append(f"msg={msg}")

        if code == 230002:
            parts.append("hint=Bot/User is not in the target chat. Add bot to the chat or use user token as a member.")

        error_obj = data.get("error")
        if isinstance(error_obj, dict):
            log_id = error_obj.get("log_id")
            if log_id:
                parts.append(f"log_id={log_id}")
        return "; ".join(parts) if parts else str(data)
