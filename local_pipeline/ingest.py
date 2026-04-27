from __future__ import annotations

from pathlib import Path
from typing import Iterable

from message_extract.extract_chat_messages import (
    build_global_mention_map,
    extract_message_text,
    format_create_time,
    load_records,
    replace_by_global_mapping,
    replace_mention_keys,
)

from .shared_types import PlainMessage


def _sender_label(sender: dict) -> str:
    if isinstance(sender.get("name"), str) and sender.get("name"):
        return str(sender["name"])
    if isinstance(sender.get("id"), str) and sender.get("id"):
        return str(sender["id"])
    return ""


def _mention_names(mentions: object) -> list[str]:
    if not isinstance(mentions, list):
        return []
    out: list[str] = []
    for item in mentions:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def ingest_messages(
    messages_file: str | Path,
    *,
    include_types: Iterable[str] = ("text", "post"),
) -> list[PlainMessage]:
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
        message_id = str(record.get("message_id", "")).strip() or f"msg-{idx}"
        chat_id = str(record.get("chat_id", "")).strip()

        output.append(
            PlainMessage(
                message_id=message_id,
                chat_id=chat_id,
                send_time=format_create_time(record.get("create_time")),
                sender=_sender_label(sender),
                mentions=_mention_names(mentions),
                content=text,
                msg_type=msg_type,
            )
        )
    return output

