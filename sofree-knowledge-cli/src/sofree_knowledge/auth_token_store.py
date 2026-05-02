from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_TOKEN_FILE


class TokenStore:
    def __init__(self, token_file: str | Path | None = None) -> None:
        self.path = Path(token_file).expanduser() if token_file else DEFAULT_TOKEN_FILE

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def save(self, token_data: dict[str, Any]) -> Path:
        payload = dict(token_data)
        saved_at = int(time.time())
        payload["saved_at"] = saved_at
        access_expires_in = _coerce_positive_int(payload.get("expires_in"))
        if access_expires_in is not None:
            payload["access_token_expires_at"] = saved_at + access_expires_in
        refresh_expires_in = _coerce_positive_int(payload.get("refresh_expires_in"))
        if refresh_expires_in is not None:
            payload["refresh_token_expires_at"] = saved_at + refresh_expires_in
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.path

    def status(self) -> dict[str, Any]:
        data = self.load()
        result = {
            "token_file": str(self.path),
            "exists": self.path.exists(),
            "has_access_token": bool(data.get("access_token")),
            "has_refresh_token": bool(data.get("refresh_token")),
            "scope": str(data.get("scope") or ""),
            "open_id": str(data.get("open_id") or ""),
            "user_id": str(data.get("user_id") or ""),
        }
        for key in ("saved_at", "access_token_expires_at", "refresh_token_expires_at", "expires_in", "refresh_expires_in"):
            if key in data:
                result[key] = data[key]
        return result


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed
