from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from .io_utils import read_json, write_json


class ChatMessageStore:
    def __init__(self, path: str | Path, *, max_messages_per_chat: int = 100) -> None:
        self.path = Path(path)
        self.max_messages_per_chat = max(1, int(max_messages_per_chat or 100))
        self._lock = RLock()

    def append(self, event: Any) -> dict[str, Any] | None:
        chat_id = str(getattr(event, "chat_id", "") or "").strip()
        if not chat_id:
            return None

        payload_fn = getattr(event, "to_dict", None)
        if not callable(payload_fn):
            return None
        message_payload = payload_fn()
        if not isinstance(message_payload, dict):
            return None
        event_payload = message_payload.get("event")
        if not isinstance(event_payload, dict):
            return None

        with self._lock:
            payload = self._load_payload()
            rows = payload.setdefault(chat_id, [])
            rows.append({"event": event_payload})
            if len(rows) > self.max_messages_per_chat:
                payload[chat_id] = rows[-self.max_messages_per_chat :]
            write_json(self.path, payload)
        return {"event": event_payload}

    def get_chat_messages(self, chat_id: str) -> list[dict[str, Any]]:
        normalized = str(chat_id or "").strip()
        if not normalized:
            return []
        with self._lock:
            payload = self._load_payload()
            rows = payload.get(normalized, [])
            return [dict(item) for item in rows if isinstance(item, dict)]

    def _load_payload(self) -> dict[str, list[dict[str, Any]]]:
        raw = read_json(self.path, default={})
        if not isinstance(raw, dict):
            return {}
        output: dict[str, list[dict[str, Any]]] = {}
        for key, value in raw.items():
            chat_id = str(key or "").strip()
            if not chat_id or not isinstance(value, list):
                continue
            rows = [
                item
                for item in value
                if isinstance(item, dict)
                and isinstance(item.get("event"), dict)
                and isinstance(item.get("event", {}).get("message"), dict)
            ]
            output[chat_id] = rows
        return output
