from __future__ import annotations

import base64
import time
import webbrowser
from typing import Any

import httpx

from .config import get_app_credentials
from .feishu_client import FeishuAPIError, MissingFeishuConfigError


DEVICE_AUTH_URL = "https://accounts.feishu.cn/open-apis/authen/v1/device_authorization"
DEVICE_TOKEN_URL = "https://open.feishu.cn/open-apis/authen/v2/oauth/token"
OFFLINE_ACCESS_SCOPE = "offline_access"


def _with_offline_access(scope: str) -> str:
    normalized = " ".join(str(scope or "").split())
    parts = [item for item in normalized.split(" ") if item]
    if OFFLINE_ACCESS_SCOPE not in parts:
        parts.append(OFFLINE_ACCESS_SCOPE)
    return " ".join(parts).strip()


def request_device_authorization(scope: str) -> dict[str, Any]:
    app_id, app_secret = get_app_credentials()
    if not app_id or not app_secret:
        raise MissingFeishuConfigError("APP_ID and SECRET_ID are required.")

    resolved_scope = _with_offline_access(scope)
    basic_auth = base64.b64encode(f"{app_id}:{app_secret}".encode("utf-8")).decode("ascii")
    try:
        response = httpx.post(
            DEVICE_AUTH_URL,
            data={"client_id": app_id, "scope": resolved_scope},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {basic_auth}",
            },
            timeout=30,
        )
    except (httpx.HTTPError, OSError) as exc:
        raise FeishuAPIError(f"Device authorization request failed: {exc}") from exc
    response.raise_for_status()
    data = response.json()
    error = str(data.get("error") or "").strip()
    if error:
        description = str(data.get("error_description") or error).strip()
        raise FeishuAPIError(f"Device authorization failed: {description}")
    verification_url = str(data.get("verification_uri_complete") or data.get("verification_uri") or "").strip()
    if not verification_url:
        raise FeishuAPIError("Device authorization response missing verification URL.")
    return {
        "device_code": str(data.get("device_code") or "").strip(),
        "user_code": str(data.get("user_code") or "").strip(),
        "verification_uri": str(data.get("verification_uri") or "").strip(),
        "verification_uri_complete": verification_url,
        "expires_in": int(data.get("expires_in") or 240),
        "interval": int(data.get("interval") or 5),
        "scope": resolved_scope,
    }


def poll_device_token(
    device_code: str,
    *,
    interval: int = 5,
    expires_in: int = 240,
) -> dict[str, Any]:
    app_id, app_secret = get_app_credentials()
    if not app_id or not app_secret:
        raise MissingFeishuConfigError("APP_ID and SECRET_ID are required.")

    deadline = time.time() + max(1, int(expires_in))
    current_interval = max(1, int(interval))
    while time.time() < deadline:
        time.sleep(current_interval)
        try:
            response = httpx.post(
                DEVICE_TOKEN_URL,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": app_id,
                    "client_secret": app_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
        except (httpx.HTTPError, OSError) as exc:
            raise FeishuAPIError(f"Device token polling failed: {exc}") from exc
        response.raise_for_status()
        data = response.json()
        error = str(data.get("error") or "").strip()
        if not error and str(data.get("access_token") or "").strip():
            return data
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            current_interval = min(current_interval + 5, 60)
            continue
        if error in {"expired_token", "invalid_grant"}:
            description = str(data.get("error_description") or "Device code expired, please retry").strip()
            raise FeishuAPIError(description)
        if error == "access_denied":
            description = str(data.get("error_description") or "Authorization denied by user").strip()
            raise FeishuAPIError(description)
        description = str(data.get("error_description") or error or "Unknown device flow error").strip()
        raise FeishuAPIError(description)
    raise TimeoutError("Timed out waiting for Feishu device authorization.")


def login_with_device_flow(
    scope: str,
    *,
    open_browser: bool = True,
) -> dict[str, Any]:
    auth_request = request_device_authorization(scope)
    verification_url = str(auth_request.get("verification_uri_complete") or "")
    if open_browser and verification_url:
        webbrowser.open(verification_url)
    token = poll_device_token(
        str(auth_request.get("device_code") or ""),
        interval=int(auth_request.get("interval") or 5),
        expires_in=int(auth_request.get("expires_in") or 240),
    )
    return {
        "request": auth_request,
        "token": token,
    }


def has_required_scopes(granted_scope: str, required_scopes: str | list[str] | tuple[str, ...]) -> bool:
    granted = {item for item in str(granted_scope or "").split() if item}
    if isinstance(required_scopes, str):
        required = {item for item in required_scopes.split() if item}
    else:
        required = {str(item).strip() for item in required_scopes if str(item).strip()}
    return required.issubset(granted)
