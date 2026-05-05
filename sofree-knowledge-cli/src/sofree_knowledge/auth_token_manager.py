from __future__ import annotations

from pathlib import Path
from typing import Any

from .auth_device_flow import has_required_scopes
from .auth_token_store import TokenStore
from .feishu_client import MissingFeishuConfigError


class TokenManager:
    def __init__(self, token_file: str | Path | None = None) -> None:
        self.store = TokenStore(token_file=token_file)

    def ensure_user_auth(self, required_scopes: str | list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
        status = self.store.status()
        if not status.get("has_access_token"):
            raise MissingFeishuConfigError(
                "Missing user access token. Run `sofree-knowledge auth-url` and `sofree-knowledge exchange-code` first."
            )
        if required_scopes and not has_required_scopes(str(status.get("scope") or ""), required_scopes):
            raise MissingFeishuConfigError(
                "User token missing required scopes. Re-authorize with `sofree-knowledge auth-url --scope ...` and `sofree-knowledge exchange-code`."
            )
        return status
