from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .feishu_client import FeishuClient, FeishuAPIError, MissingFeishuConfigError


def collect_messages(
    client: FeishuClient,
    output_dir: str | Path,
    output_subdir: str = "",
    chat_ids: str | list[str] | None = None,
    include_visible_chats: bool = True,
    start_time: str = "",
    end_time: str = "",
    max_chats: int = 1000,
    max_messages_per_chat: int = 1000,
    page_size: int = 50,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = root / "message_archive" / (output_subdir or run_id)
    archive_dir.mkdir(parents=True, exist_ok=True)

    chats_by_id: dict[str, dict[str, Any]] = {}
    if include_visible_chats:
        for chat in list_visible_chats(client, max_items=max_chats):
            chat_id = str(chat.get("chat_id", ""))
            if chat_id:
                chats_by_id[chat_id] = dict(chat)
    for chat_id in split_chat_ids(chat_ids):
        chats_by_id.setdefault(chat_id, {"chat_id": chat_id, "name": "", "chat_mode": "unknown"})

    chat_records = list(chats_by_id.values())[:max_chats]
    chats_path = archive_dir / "chats.jsonl"
    messages_path = archive_dir / "messages.jsonl"
    manifest_path = archive_dir / "manifest.json"

    with chats_path.open("w", encoding="utf-8") as chat_file:
        for chat in chat_records:
            chat_file.write(json.dumps(normalize_chat_record(chat), ensure_ascii=False) + "\n")

    total_messages = 0
    collected_at = datetime.now(timezone.utc).isoformat()
    with messages_path.open("w", encoding="utf-8") as message_file:
        for chat in chat_records:
            chat_id = str(chat.get("chat_id", ""))
            try:
                messages = list_chat_messages(
                    client,
                    chat_id=chat_id,
                    start_time=start_time,
                    end_time=end_time,
                    max_items=max_messages_per_chat,
                    page_size=page_size,
                )
                for message in messages:
                    message["collected_at"] = collected_at
                    message["chat"] = normalize_chat_record(chat)
                    message_file.write(json.dumps(message, ensure_ascii=False) + "\n")
                total_messages += len(messages)
            except (FeishuAPIError, MissingFeishuConfigError):
                # 跳过无权限或配置错误的群，不影响整体归档流程
                continue

    manifest = {
        "run_id": run_id,
        "archive_dir": str(archive_dir),
        "start_time": start_time,
        "end_time": end_time,
        "max_chats": max_chats,
        "max_messages_per_chat": max_messages_per_chat,
        "page_size": page_size,
        "include_visible_chats": include_visible_chats,
        "chat_count": len(chat_records),
        "message_count": total_messages,
        "files": {
            "manifest": str(manifest_path),
            "chats": str(chats_path),
            "messages": str(messages_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def list_visible_chats(client: FeishuClient, max_items: int = 1000, page_size: int = 100) -> list[dict[str, Any]]:
    chats: list[dict[str, Any]] = []
    page_token = ""
    while len(chats) < max_items:
        page = client.list_visible_chats(page_size=min(page_size, max_items - len(chats)), page_token=page_token)
        chats.extend(page.get("items", []))
        page_token = str(page.get("page_token", "") or "")
        if not page.get("has_more") or not page_token:
            break
    return chats[:max_items]


def list_chat_messages(
    client: FeishuClient,
    chat_id: str,
    start_time: str = "",
    end_time: str = "",
    max_items: int = 1000,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    page_token = ""
    start_ts = normalize_feishu_time(start_time)
    end_ts = normalize_feishu_time(end_time, end_of_day=True)
    while len(messages) < max_items:
        page = client.list_chat_messages(
            chat_id=chat_id,
            start_time=start_ts,
            end_time=end_ts,
            page_size=min(page_size, max_items - len(messages)),
            page_token=page_token,
            sort="asc",
        )
        messages.extend(
            normalize_chat_message(item, fallback_chat_id=chat_id)
            for item in page.get("items", [])
            if not _is_filtered_card_message(item)
        )
        page_token = str(page.get("page_token", "") or "")
        if not page.get("has_more") or not page_token:
            break
    return messages[:max_items]


def normalize_chat_record(chat: dict[str, Any]) -> dict[str, Any]:
    return {
        "chat_id": str(chat.get("chat_id", "")),
        "name": str(chat.get("name", "")),
        "chat_mode": str(chat.get("chat_mode") or chat.get("chat_type") or ""),
        "chat_type": str(chat.get("chat_type", "")),
        "owner_id": str(chat.get("owner_id", "")),
        "external": bool(chat.get("external", False)),
        "raw": chat,
    }


def normalize_chat_message(item: dict[str, Any], fallback_chat_id: str = "") -> dict[str, Any]:
    body = item.get("body", {})
    raw_content = ""
    if isinstance(body, dict):
        raw_content = str(body.get("content", "") or "")
    elif isinstance(body, str):
        raw_content = body
    sender = item.get("sender", {})
    if not isinstance(sender, dict):
        sender = {}
    message_id = str(item.get("message_id", ""))
    chat_id = str(item.get("chat_id") or fallback_chat_id)
    return {
        "message_id": message_id,
        "chat_id": chat_id,
        "thread_id": str(item.get("thread_id", "")),
        "root_id": str(item.get("root_id", "")),
        "parent_id": str(item.get("parent_id", "")),
        "msg_type": str(item.get("msg_type", "")),
        "create_time": str(item.get("create_time", "")),
        "update_time": str(item.get("update_time", "")),
        "deleted": bool(item.get("deleted", False)),
        "updated": bool(item.get("updated", False)),
        "sender": sender,
        "content": extract_message_text(raw_content),
        "raw_content": raw_content,
        "mentions": item.get("mentions", []) if isinstance(item.get("mentions", []), list) else [],
        "message_url": build_message_url(chat_id=chat_id, message_id=message_id),
        "raw": item,
    }


def _is_filtered_card_message(item: dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    return str(item.get("card_msg_content_type") or "").strip().lower() == "raw_card_content"


def build_message_url(chat_id: str, message_id: str) -> str:
    normalized_chat_id = str(chat_id or "").strip()
    normalized_message_id = str(message_id or "").strip()
    if not normalized_chat_id or not normalized_message_id:
        return ""
    encoded_chat_id = quote(normalized_chat_id, safe="")
    encoded_message_id = quote(normalized_message_id, safe="")
    return (
        "https://applink.feishu.cn/client/chat/open?"
        f"chatId={encoded_chat_id}&openChatId={encoded_chat_id}&openMessageId={encoded_message_id}"
    )


def extract_message_text(raw_content: str) -> str:
    if not raw_content:
        return ""
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return raw_content
    if isinstance(payload, dict):
        for key in ("text", "title", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return raw_content


def split_chat_ids(chat_ids: str | list[str] | None) -> list[str]:
    if chat_ids is None:
        return []
    values = chat_ids.split(",") if isinstance(chat_ids, str) else chat_ids
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        chat_id = str(value).strip()
        if chat_id and chat_id not in seen:
            seen.add(chat_id)
            result.append(chat_id)
    return result


def normalize_feishu_time(value: str, end_of_day: bool = False) -> str:
    value = str(value or "").strip()
    if not value or value.isdigit():
        return value
    try:
        if len(value) == 10 and value[4] == "-" and value[7] == "-":
            suffix = "T23:59:59+00:00" if end_of_day else "T00:00:00+00:00"
            dt = datetime.fromisoformat(value + suffix)
        else:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return str(int(dt.timestamp()))
