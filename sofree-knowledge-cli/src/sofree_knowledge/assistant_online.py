from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote
from urllib.parse import urlparse

from .archive import list_chat_messages, list_visible_chats, split_chat_ids
from .config import get_user_identity
from .feishu_client import FeishuClient, FeishuAPIError

_DOC_HOST_HINTS = ("feishu.cn", "larksuite.com")
_DOC_PATH_HINTS = ("/docx/", "/wiki/", "/base/", "/sheet/", "/slides/", "/file/")
_MESSAGE_NOISE_PATTERNS = (
    "llm 调用失败",
    "too many requests",
    "llm 配置不完整",
    "缺少:",
    "[plan_command]",
    "write_policy:",
    "output_format:",
    "我没办法直接帮你",
    "可以和我说说具体情况",
    '"json_card"',
    "{\\\"json_card\\\"",
)


def collect_online_personal_inputs(
    client: FeishuClient,
    target_user_id: str = "",
    token_file: str = "",
    chat_ids: str | list[str] | None = None,
    include_visible_chats: bool = True,
    max_chats: int = 20,
    max_messages_per_chat: int = 200,
    max_drive_docs: int = 50,
    max_knowledge: int = 30,
    recent_days: int = 7,
) -> dict[str, Any]:
    resolved_target = str(target_user_id or "").strip()
    if not resolved_target:
        identity = get_user_identity(token_file=token_file or None)
        resolved_target = str(identity.get("open_id") or identity.get("user_id") or "").strip()

    chats_by_id: dict[str, dict[str, Any]] = {}
    if include_visible_chats:
        for chat in list_visible_chats(client, max_items=max_chats):
            chat_id = str(chat.get("chat_id") or "").strip()
            if chat_id:
                chats_by_id[chat_id] = chat

    for chat_id in split_chat_ids(chat_ids):
        chats_by_id.setdefault(chat_id, {"chat_id": chat_id, "name": "", "chat_mode": "unknown"})

    chat_records = list(chats_by_id.values())[:max_chats]

    messages: list[dict[str, Any]] = []
    for chat in chat_records:
        chat_id = str(chat.get("chat_id") or "").strip()
        if not chat_id:
            continue
        chat_messages = list_chat_messages(
            client,
            chat_id=chat_id,
            max_items=max_messages_per_chat,
            page_size=min(50, max_messages_per_chat),
        )
        for msg in chat_messages:
            text = _normalize_message_text(msg.get("content", ""))
            if _is_noise_message_text(text):
                continue
            if not _is_within_recent_days(str(msg.get("create_time") or ""), recent_days):
                continue
            messages.append(
                {
                    "message_id": msg.get("message_id", ""),
                    "chat_id": msg.get("chat_id", chat_id),
                    "sender_name": _extract_sender_name(msg.get("sender", {})),
                    "content": text,
                    "text": text,
                    "create_time": msg.get("create_time", ""),
                    "sender": msg.get("sender", {}),
                    "message_url": _resolve_online_message_url(
                        message=msg,
                    ),
                }
            )

    drive_docs: list[dict[str, Any]] = []
    drive_error = ""
    try:
        drive_docs = list_recent_drive_docs(client, max_items=max_drive_docs, recent_days=recent_days)
    except FeishuAPIError as exc:
        drive_error = str(exc)

    linked_docs, link_access_records = extract_docs_and_access_from_messages(messages, resolved_target)

    # 批量获取消息提及文档的真实标题（使用独立实现，不依赖FeishuClient内置方法）
    for doc in linked_docs:
        try:
            doc_token = doc["doc_id"]
            url = doc.get("url", "")
            doc_type = "docx"
            if "/wiki/" in url:
                doc_type = "wiki"
            elif "/base/" in url:
                doc_type = "base"
            elif "/sheet/" in url:
                doc_type = "sheet"
            elif "/slides/" in url:
                doc_type = "slides"
            elif "/file/" in url:
                doc_type = "file"

            # 调用本地独立实现的元数据查询函数
            meta = _get_doc_meta(client, doc_token, doc_type=doc_type)
            if meta.get("title"):
                doc["title"] = meta["title"]
                if meta.get("updated_at"):
                    doc["updated_at"] = meta["updated_at"]
        except Exception:
            # 查询失败不影响主流程，保持原有标题即可
            pass

    documents = _merge_documents(drive_docs, linked_docs)
    access_records = link_access_records
    knowledge_items = build_knowledge_items(messages=messages, documents=documents, max_items=max_knowledge)

    return {
        "documents": documents,
        "access_records": access_records,
        "messages": messages,
        "knowledge_items": knowledge_items,
        "resolved_target_user_id": resolved_target,
        "meta": {
            "chat_count": len(chat_records),
            "message_count": len(messages),
            "document_count": len(documents),
            "recent_days": recent_days,
            "drive_error": drive_error,
        },
    }


def list_recent_drive_docs(client: FeishuClient, max_items: int = 50, recent_days: int = 7) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    page_token = ""
    while len(docs) < max_items:
        page = client.list_drive_files(page_size=min(50, max_items - len(docs)), page_token=page_token)
        items = page.get("items", []) if isinstance(page, dict) else []
        for item in items:
            updated_at = str(item.get("modified_time") or item.get("edit_time") or "")
            if not _is_within_recent_days(updated_at, recent_days):
                continue
            docs.append(
                {
                    "doc_id": str(item.get("token") or item.get("file_token") or item.get("id") or ""),
                    "title": str(item.get("name") or item.get("title") or ""),
                    "summary": str(item.get("name") or item.get("title") or ""),
                    "url": str(item.get("url") or item.get("link") or ""),
                    "updated_at": updated_at,
                    "source": "drive",
                }
            )
        page_token = str(page.get("page_token", "") or "")
        if not page.get("has_more") or not page_token:
            break
    return docs[:max_items]


def extract_docs_and_access_from_messages(
    messages: list[dict[str, Any]],
    target_user_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    docs_by_key: dict[str, dict[str, Any]] = {}
    access_counts: Counter[tuple[str, str]] = Counter()

    for message in messages:
        text = str(message.get("text") or message.get("content") or "")
        if not text:
            continue
        urls = _extract_urls(text)
        sender_id = _extract_sender_id(message)
        for url in urls:
            if not _looks_like_doc_url(url):
                continue
            key = _doc_key_from_url(url)
            docs_by_key.setdefault(
                key,
                {
                    "doc_id": key,
                    "title": "相关文档",
                    "summary": _trim(text, 120),
                    "url": url,
                    "updated_at": str(message.get("create_time") or ""),
                    "source": "message_link",
                },
            )
            if sender_id and target_user_id and sender_id == target_user_id:
                access_counts[(key, sender_id)] += 1

    access_records = [
        {
            "doc_id": doc_id,
            "user_id": user_id,
            "action": "view",
            "count": count,
        }
        for (doc_id, user_id), count in access_counts.items()
    ]
    return list(docs_by_key.values()), access_records


def build_knowledge_items(
    messages: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    max_items: int = 30,
) -> list[dict[str, Any]]:
    freq: Counter[str] = Counter()
    for message in messages:
        text = str(message.get("text") or message.get("content") or "")
        for token in _extract_keywords(text):
            freq[token] += 1

    knowledge: list[dict[str, Any]] = []
    for idx, (token, count) in enumerate(freq.most_common(max_items), start=1):
        related = [doc for doc in documents if token in str(doc.get("title", "")) or token in str(doc.get("summary", ""))]
        knowledge.append(
            {
                "id": f"kw_{idx}",
                "title": token,
                "content": f"近期消息中出现 {count} 次，关联文档 {len(related)} 篇。",
            }
        )
    return knowledge


def _merge_documents(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in [*primary, *secondary]:
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        if doc_id not in merged:
            merged[doc_id] = dict(item)
            continue
        current = merged[doc_id]
        if not current.get("url") and item.get("url"):
            current["url"] = item["url"]
        if not current.get("updated_at") and item.get("updated_at"):
            current["updated_at"] = item["updated_at"]
        if len(str(item.get("summary") or "")) > len(str(current.get("summary") or "")):
            current["summary"] = item.get("summary", "")
        if len(str(item.get("title") or "")) > len(str(current.get("title") or "")):
            current["title"] = item.get("title", "")
    return list(merged.values())


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s\]>\)\"]+", text)


def _looks_like_doc_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    return any(hint in host for hint in _DOC_HOST_HINTS) and any(hint in path for hint in _DOC_PATH_HINTS)


def _doc_key_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in (parsed.path or "").split("/") if part]
    if not parts:
        return url
    return parts[-1]


def _extract_sender_id(message: dict[str, Any]) -> str:
    sender = message.get("sender", {})
    if isinstance(sender, dict):
        sender_id = sender.get("sender_id")
        if isinstance(sender_id, dict):
            for key in ("open_id", "user_id", "union_id"):
                value = sender_id.get(key)
                if value:
                    return str(value)
        for key in ("id", "open_id", "user_id"):
            value = sender.get(key)
            if value:
                return str(value)
    return ""


def _extract_keywords(text: str) -> set[str]:
    normalized = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    result: set[str] = set()
    for token in tokens:
        if token.strip():
            result.add(token)
            if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
                for idx in range(0, len(token) - 1):
                    result.add(token[idx : idx + 2])
    return result


def _trim(text: str, max_len: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def _normalize_message_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _is_noise_message_text(text: str) -> bool:
    lowered = str(text or "").lower().strip()
    if not lowered:
        return True
    if _looks_like_bot_prompt(lowered):
        return True
    if lowered.startswith("command:"):
        return True
    if lowered.startswith("[plan_command]"):
        return True
    return any(pattern in lowered for pattern in _MESSAGE_NOISE_PATTERNS)


def _looks_like_bot_prompt(lowered: str) -> bool:
    if lowered.startswith("@sofree") and ("请生成" in lowered or "要求如下" in lowered):
        return True
    if lowered.startswith("@_user_") and ("可以从几个维度讨论" in lowered or "口味偏好" in lowered):
        return True
    if lowered.startswith("@") and "1." in lowered and "2." in lowered and "3." in lowered and len(lowered) > 80:
        return True
    return False


def _is_within_recent_days(raw: str, recent_days: int) -> bool:
    days = max(1, int(recent_days or 7))
    value = str(raw or "").strip()
    if not value:
        return True
    now = datetime.now(timezone.utc)
    dt: datetime | None = None
    if value.isdigit():
        ts = int(value)
        if ts > 10_000_000_000:
            ts = ts // 1000
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            return True
    else:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return True
    return (now - dt).total_seconds() <= days * 24 * 3600


def _resolve_online_message_url(
    message: dict[str, Any],
) -> str:
    for key in ("message_url", "open_message_url", "deep_link", "link"):
        value = str(message.get(key) or "").strip()
        if value.startswith("http://") or value.startswith("https://"):
            return value
    chat_id = str(message.get("chat_id") or "").strip()
    message_id = str(message.get("message_id") or "").strip()
    if not chat_id or not message_id:
        return ""
    encoded_chat_id = quote(chat_id, safe="")
    encoded_message_id = quote(message_id, safe="")
    return (
        "https://applink.feishu.cn/client/chat/open?"
        f"chatId={encoded_chat_id}&openChatId={encoded_chat_id}&openMessageId={encoded_message_id}"
    )


def _get_doc_meta(client: FeishuClient, doc_token: str, doc_type: str = "docx") -> dict[str, Any]:
    """独立的文档元数据查询功能，不依赖FeishuClient的内置方法"""
    doc_type_map = {
        "docx": "docs",
        "wiki": "wiki",
        "base": "bitable",
        "sheet": "sheets",
        "slides": "slides",
        "file": "drive",
    }
    api_type = doc_type_map.get(doc_type, "docs")
    path = f"/open-apis/{api_type}/v1/{doc_type}s/{doc_token}"
    if doc_type == "file":
        path = f"/open-apis/drive/v1/files/{doc_token}"

    try:
        # 使用用户token查询
        data = client.request("GET", path)
    except FeishuAPIError:
        try:
            # 失败则用租户token重试
            data = client.request(
                "GET",
                path,
                access_token=client.get_tenant_access_token(),
            )
        except Exception:
            return {}

    body = data.get("data", data)
    if doc_type == "file":
        return {
            "title": body.get("name", ""),
            "url": body.get("url", ""),
            "updated_at": body.get("modified_time", ""),
        }
    return {
        "title": body.get("title", ""),
        "url": body.get("url", ""),
        "updated_at": body.get("modified_time", ""),
    }


def _extract_sender_name(sender: Any) -> str:
    if not isinstance(sender, dict):
        return ""
    for key in ("name", "display_name", "sender_name", "user_name", "nickname"):
        value = str(sender.get(key) or "").strip()
        if value:
            return value
    sender_id = sender.get("sender_id")
    if isinstance(sender_id, dict):
        for key in ("open_id", "user_id", "union_id"):
            value = str(sender_id.get(key) or "").strip()
            if value:
                return value
    for key in ("id", "open_id", "user_id"):
        value = str(sender.get(key) or "").strip()
        if value:
            return value
    return ""
