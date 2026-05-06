from __future__ import annotations

from typing import Any

from .feishu_client import FeishuAPIError


DEVICE_AUTH_URL = "https://accounts.feishu.cn/open-apis/authen/v1/device_authorization"
DEVICE_TOKEN_URL = "https://open.feishu.cn/open-apis/authen/v2/oauth/token"
OFFLINE_ACCESS_SCOPE = "offline_access"
DEVICE_FLOW_DISABLED_MESSAGE = (
    "Feishu device flow authorization is disabled in this project. "
    "Use `sofree-knowledge auth-url` and `sofree-knowledge exchange-code` instead."
)


def _with_offline_access(scope: str) -> str:
    normalized = " ".join(str(scope or "").split())
    parts = [item for item in normalized.split(" ") if item]
    if OFFLINE_ACCESS_SCOPE not in parts:
        parts.append(OFFLINE_ACCESS_SCOPE)
    return " ".join(parts).strip()


def request_device_authorization(scope: str) -> dict[str, Any]:
    raise FeishuAPIError(DEVICE_FLOW_DISABLED_MESSAGE)


def poll_device_token(
    device_code: str,
    *,
    interval: int = 5,
    expires_in: int = 240,
) -> dict[str, Any]:
    raise FeishuAPIError(DEVICE_FLOW_DISABLED_MESSAGE)


def login_with_device_flow(
    scope: str,
    *,
    open_browser: bool = True,
) -> dict[str, Any]:
    raise FeishuAPIError(DEVICE_FLOW_DISABLED_MESSAGE)


def has_required_scopes(granted_scope: str, required_scopes: str | list[str] | tuple[str, ...]) -> bool:
    granted = {item for item in str(granted_scope or "").split() if item}
    if isinstance(required_scopes, str):
        required = {item for item in required_scopes.split() if item}
    else:
        required = {str(item).strip() for item in required_scopes if str(item).strip()}
    return required.issubset(granted)
