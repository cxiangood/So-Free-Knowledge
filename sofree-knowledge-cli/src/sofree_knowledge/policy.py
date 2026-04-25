from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_SCOPES = {"global_review", "chat_only"}


class KnowledgePolicyStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.path = Path(output_dir).expanduser() / "knowledge_policy.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"default_scope": "global_review", "chats": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        chats = data.get("chats")
        if not isinstance(chats, dict):
            chats = {}
        default_scope = data.get("default_scope")
        if default_scope not in VALID_SCOPES:
            default_scope = "global_review"
        return {"default_scope": default_scope, "chats": chats}

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_scope(self, chat_id: str) -> dict[str, Any]:
        data = self.load()
        chat = data["chats"].get(chat_id, {})
        scope = chat.get("scope") if isinstance(chat, dict) else None
        return {
            "chat_id": chat_id,
            "scope": scope if scope in VALID_SCOPES else data["default_scope"],
            "policy_file": str(self.path),
        }

    def set_scope(self, chat_id: str, scope: str) -> dict[str, Any]:
        if scope not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(VALID_SCOPES)}")
        if not chat_id:
            raise ValueError("chat_id is required")
        data = self.load()
        chats = data["chats"]
        chat = chats.get(chat_id)
        if not isinstance(chat, dict):
            chat = {}
        chat["scope"] = scope
        chats[chat_id] = chat
        self.save(data)
        return {"chat_id": chat_id, "scope": scope, "policy_file": str(self.path)}
