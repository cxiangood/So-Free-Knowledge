from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class _TokenCache:
    token: str = ""
    expires_at: float = 0.0


_TOKEN_CACHE = _TokenCache()
_USER_NAME_CACHE: dict[str, str] = {}


def get_tenant_access_token(app_id: str, app_secret: str, timeout: float = 5.0) -> str:
    now = time.time()
    if _TOKEN_CACHE.token and now < _TOKEN_CACHE.expires_at - 30:
        return _TOKEN_CACHE.token
    resp = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json() if resp.content else {}
    token = str(payload.get("tenant_access_token", "")).strip()
    expire = int(payload.get("expire", 0) or 0)
    if not token:
        return ""
    _TOKEN_CACHE.token = token
    _TOKEN_CACHE.expires_at = now + max(60, expire)
    return token


def get_user_name_by_user_id(user_id: str, app_id: str, app_secret: str, timeout: float = 5.0) -> str:
    uid = str(user_id or "").strip()
    if not uid:
        return ""
    if uid in _USER_NAME_CACHE:
        return _USER_NAME_CACHE[uid]
    token = get_tenant_access_token(app_id, app_secret, timeout=timeout)
    if not token:
        return ""
    resp = requests.get(
        f"https://open.feishu.cn/open-apis/contact/v3/users/{uid}",
        params={"user_id_type": "user_id"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json() if resp.content else {}
    name = str(payload.get("data", {}).get("user", {}).get("name", "")).strip()
    if name:
        _USER_NAME_CACHE[uid] = name
    return name


__all__ = ["get_tenant_access_token", "get_user_name_by_user_id"]
