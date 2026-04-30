from __future__ import annotations

import json
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(token_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.path

    def status(self) -> dict[str, Any]:
        data = self.load()
        return {
            "token_file": str(self.path),
            "exists": self.path.exists(),
            "has_access_token": bool(data.get("access_token")),
            "has_refresh_token": bool(data.get("refresh_token")),
            "scope": str(data.get("scope") or ""),
            "open_id": str(data.get("open_id") or ""),
            "user_id": str(data.get("user_id") or ""),
        }
