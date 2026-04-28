from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PlainMessage:
    message_id: str
    chat_id: str
    send_time: str
    sender: str
    mentions: list[str]
    content: str
    msg_type: str = "text"
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MessageEvent:
    event_type: str
    event_id: str
    create_time: str
    message_id: str
    chat_id: str
    chat_type: str
    message_type: str
    content_text: str
    content_raw: str
    root_id: str
    parent_id: str
    update_time: str
    thread_id: str
    sender_open_id: str
    sender_union_id: str
    sender_user_id: str
    sender_type: str
    tenant_key: str
    mentions: list[dict[str, Any]] = field(default_factory=list)
    user_agent: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.raw, dict) and isinstance(self.raw.get("event"), dict):
            return {"event": self.raw["event"]}
        return {
            "event": {
                "sender": {
                    "sender_id": {
                        "union_id": self.sender_union_id,
                        "user_id": self.sender_user_id,
                        "open_id": self.sender_open_id,
                    },
                    "sender_type": self.sender_type,
                    "tenant_key": self.tenant_key,
                },
                "message": {
                    "message_id": self.message_id,
                    "root_id": self.root_id,
                    "parent_id": self.parent_id,
                    "create_time": self.create_time,
                    "update_time": self.update_time,
                    "chat_id": self.chat_id,
                    "thread_id": self.thread_id,
                    "chat_type": self.chat_type,
                    "message_type": self.message_type,
                    "content": self.content_raw,
                    "mentions": self.mentions,
                    "user_agent": self.user_agent,
                },
            }
        }


__all__ = ["MessageEvent", "PlainMessage"]

