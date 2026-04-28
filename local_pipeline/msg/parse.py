from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from message_extract.extract_chat_messages import (
    build_global_mention_map,
    extract_message_text,
    format_create_time,
    load_records,
    replace_by_global_mapping,
    replace_mention_keys,
)

from .types import MessageEvent, PlainMessage


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_content_text(raw_content: Any) -> str:
    content = _safe_text(raw_content)
    if not content:
        return ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return _safe_text(payload.get("text"))
    return content


def _mentions_to_names(mentions: Any) -> list[str]:
    if not isinstance(mentions, list):
        return []
    output: list[str] = []
    for item in mentions:
        if not isinstance(item, dict):
            continue
        name = _safe_text(item.get("name", ""))
        if name:
            output.append(name)
    return output


def _get_field(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _sender_id_dict(sender_id: Any) -> dict[str, str]:
    return {
        "union_id": _safe_text(_get_field(sender_id, "union_id", "")),
        "user_id": _safe_text(_get_field(sender_id, "user_id", "")),
        "open_id": _safe_text(_get_field(sender_id, "open_id", "")),
    }


def _mentions_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        mention_id = _sender_id_dict(item.get("id", {}))
        rows.append(
            {
                "key": _safe_text(item.get("key", "")),
                "id": mention_id,
                "mentioned_type": _safe_text(item.get("mentioned_type", "")),
                "name": _safe_text(item.get("name", "")),
                "tenant_key": _safe_text(item.get("tenant_key", "")),
            }
        )
    return rows


def event_row_to_plain_message(row: dict[str, Any]) -> PlainMessage | None:
    event = row.get("event")
    if not isinstance(event, dict):
        return None
    sender = event.get("sender")
    message = event.get("message")
    if not isinstance(sender, dict) or not isinstance(message, dict):
        return None
    sender_id = sender.get("sender_id")
    if not isinstance(sender_id, dict):
        sender_id = {}

    message_id = _safe_text(message.get("message_id", ""))
    chat_id = _safe_text(message.get("chat_id", ""))
    if not message_id or not chat_id:
        return None
    content_raw = _safe_text(message.get("content", ""))
    return PlainMessage(
        message_id=message_id,
        chat_id=chat_id,
        send_time=_safe_text(message.get("create_time", "")),
        sender=_safe_text(sender_id.get("open_id", "")),
        mentions=_mentions_to_names(message.get("mentions", [])),
        content=_parse_content_text(content_raw),
        msg_type=_safe_text(message.get("message_type", "")) or "text",
    )


def parse_message_event(data: Any, supported_event_type: str = "im.message.receive_v1") -> MessageEvent | None:
    header = _get_field(data, "header", None)
    event = _get_field(data, "event", None)
    if event is None:
        return None
    message = _get_field(event, "message", None)
    sender = _get_field(event, "sender", None)
    if message is None or sender is None:
        return None

    event_type = _safe_text(_get_field(header, "event_type", "")) or supported_event_type
    if event_type and event_type != supported_event_type:
        return None
    sender_id = _get_field(sender, "sender_id", None)
    raw = {
        "event": {
            "sender": {
                "sender_id": _sender_id_dict(sender_id),
                "sender_type": _safe_text(_get_field(sender, "sender_type", "")),
                "tenant_key": _safe_text(_get_field(sender, "tenant_key", "")),
            },
            "message": {
                "message_id": _safe_text(_get_field(message, "message_id", "")),
                "root_id": _safe_text(_get_field(message, "root_id", "")),
                "parent_id": _safe_text(_get_field(message, "parent_id", "")),
                "create_time": _safe_text(_get_field(message, "create_time", "")),
                "update_time": _safe_text(_get_field(message, "update_time", "")),
                "chat_id": _safe_text(_get_field(message, "chat_id", "")),
                "thread_id": _safe_text(_get_field(message, "thread_id", "")),
                "chat_type": _safe_text(_get_field(message, "chat_type", "")),
                "message_type": _safe_text(_get_field(message, "message_type", "")),
                "content": _safe_text(_get_field(message, "content", "")),
                "mentions": _mentions_list(_get_field(message, "mentions", [])),
                "user_agent": _safe_text(_get_field(message, "user_agent", "")),
            },
        }
    }
    return MessageEvent(
        event_type=event_type,
        event_id=_safe_text(_get_field(header, "event_id", "")),
        create_time=_safe_text(_get_field(header, "create_time", "")) or raw["event"]["message"]["create_time"],
        message_id=raw["event"]["message"]["message_id"],
        chat_id=raw["event"]["message"]["chat_id"],
        chat_type=raw["event"]["message"]["chat_type"],
        message_type=raw["event"]["message"]["message_type"],
        content_text=_parse_content_text(raw["event"]["message"]["content"]),
        content_raw=raw["event"]["message"]["content"],
        root_id=raw["event"]["message"]["root_id"],
        parent_id=raw["event"]["message"]["parent_id"],
        update_time=raw["event"]["message"]["update_time"],
        thread_id=raw["event"]["message"]["thread_id"],
        sender_open_id=raw["event"]["sender"]["sender_id"]["open_id"],
        sender_union_id=raw["event"]["sender"]["sender_id"]["union_id"],
        sender_user_id=raw["event"]["sender"]["sender_id"]["user_id"],
        sender_type=raw["event"]["sender"]["sender_type"],
        tenant_key=raw["event"]["sender"]["tenant_key"],
        mentions=raw["event"]["message"]["mentions"],
        user_agent=raw["event"]["message"]["user_agent"],
        raw=raw,
    )


def should_accept_message_event(event: MessageEvent) -> bool:
    if event.sender_type != "user":
        return False
    return event.chat_type in {"p2p", "group"}


def load_plain_messages_from_archive(messages_file: str | Path, include_types: tuple[str, ...] = ("text", "post")) -> list[PlainMessage]:
    path = Path(messages_file)
    records = load_records(path)
    include = {str(item).strip().lower() for item in include_types if str(item).strip()}
    mention_map = build_global_mention_map(records)
    output: list[PlainMessage] = []
    for idx, record in enumerate(records):
        msg_type = str(record.get("msg_type", "")).lower()
        if msg_type not in include:
            continue
        text = extract_message_text(record).strip()
        if not text:
            continue
        mentions_raw = record.get("mentions")
        mentions = [m for m in mentions_raw if isinstance(m, dict)] if isinstance(mentions_raw, list) else []
        text = replace_mention_keys(text, mentions)
        text = replace_by_global_mapping(text, mention_map).strip()
        if not text:
            continue
        sender_raw = record.get("sender")
        sender = sender_raw if isinstance(sender_raw, dict) else {}
        sender_label = str(sender.get("name") or sender.get("id") or "")
        message_id = str(record.get("message_id", "")).strip() or f"msg-{idx}"
        chat_id = str(record.get("chat_id", "")).strip()
        output.append(
            PlainMessage(
                message_id=message_id,
                chat_id=chat_id,
                send_time=format_create_time(record.get("create_time")),
                sender=sender_label,
                mentions=_mentions_to_names(mentions),
                content=text,
                msg_type=msg_type,
            )
        )
    return output


def plain_message_to_event(message: PlainMessage) -> MessageEvent:
    raw = {
        "event": {
            "sender": {
                "sender_id": {"union_id": "", "user_id": "", "open_id": message.sender},
                "sender_type": "user",
                "tenant_key": "",
            },
            "message": {
                "message_id": message.message_id,
                "root_id": "",
                "parent_id": "",
                "create_time": message.send_time,
                "update_time": message.send_time,
                "chat_id": message.chat_id,
                "thread_id": "",
                "chat_type": "group",
                "message_type": message.msg_type or "text",
                "content": json.dumps({"text": message.content}, ensure_ascii=False),
                "mentions": [{"name": item} for item in message.mentions],
                "user_agent": "",
            },
        }
    }
    return MessageEvent(
        event_type="im.message.receive_v1",
        event_id=f"evt-{message.message_id}",
        create_time=message.send_time,
        message_id=message.message_id,
        chat_id=message.chat_id,
        chat_type="group",
        message_type=message.msg_type or "text",
        content_text=message.content,
        content_raw=raw["event"]["message"]["content"],
        root_id="",
        parent_id="",
        update_time=message.send_time,
        thread_id="",
        sender_open_id=message.sender,
        sender_union_id="",
        sender_user_id="",
        sender_type="user",
        tenant_key="",
        mentions=raw["event"]["message"]["mentions"],
        user_agent="",
        raw=raw,
    )


__all__ = [
    "event_row_to_plain_message",
    "plain_message_to_event",
    "load_plain_messages_from_archive",
    "parse_message_event",
    "should_accept_message_event",
]
