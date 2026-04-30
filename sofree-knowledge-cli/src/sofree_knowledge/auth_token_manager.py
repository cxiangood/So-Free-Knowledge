from __future__ import annotations

from pathlib import Path
from typing import Any

from .auth_device_flow import has_required_scopes, login_with_device_flow, poll_device_token, request_device_authorization
from .auth_token_store import TokenStore
from .feishu_client import MissingFeishuConfigError


class TokenManager:
    def __init__(self, token_file: str | Path | None = None) -> None:
        self.store = TokenStore(token_file=token_file)

    def ensure_user_auth(self, required_scopes: str | list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
        status = self.store.status()
        if not status.get("has_access_token"):
            raise MissingFeishuConfigError("Missing user access token. Run `sofree-knowledge auth login` first.")
        if required_scopes and not has_required_scopes(str(status.get("scope") or ""), required_scopes):
            raise MissingFeishuConfigError("User token missing required scopes. Re-run `sofree-knowledge auth login --scope ...`.")
        return status

    def start_device_login(self, scope: str) -> dict[str, Any]:
        return {"flow": "device_code", "request": request_device_authorization(scope)}

    def resume_device_login(self, device_code: str, *, interval: int = 5, expires_in: int = 240) -> dict[str, Any]:
        token = poll_device_token(device_code, interval=interval, expires_in=expires_in)
        self.store.save(token)
        return {"flow": "device_code", "token": token}

    def device_login(self, scope: str, *, open_browser: bool = True) -> dict[str, Any]:
        result = login_with_device_flow(scope, open_browser=open_browser)
        token = result.get("token", {})
        if isinstance(token, dict):
            self.store.save(token)
        return {"flow": "device_code", **result}
