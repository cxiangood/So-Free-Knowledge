from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote
from urllib.parse import urlparse

from .archive import list_chat_messages, list_visible_chats, split_chat_ids
from .config import get_user_identity
from .feishu_client import FeishuClient, FeishuAPIError, MissingFeishuConfigError
from .interest_filter import apply_interest_filter_annotations

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
    max_behavior_docs: int = 200,
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

    # 收集用户ID到昵称的映射
    user_id_to_name: dict[str, str] = {}
    target_aliases = _build_target_aliases(client, resolved_target, user_id_to_name)

    messages: list[dict[str, Any]] = []
    all_messages: list[dict[str, Any]] = []
    for chat in chat_records:
        chat_id = str(chat.get("chat_id") or "").strip()
        if not chat_id:
            continue
        try:
            chat_messages = list_chat_messages(
                client,
                chat_id=chat_id,
                max_items=max_messages_per_chat,
                page_size=min(50, max_messages_per_chat),
            )
            for msg in chat_messages:
                if _should_skip_message(msg):
                    continue
                # 收集发件人信息
                sender = msg.get("sender", {})
                sender_name = _extract_sender_name(sender)
                sender_id = _extract_sender_id(msg)

                # 确保发件人名称是真实昵称，不是ID
                if sender_id and (not sender_name or _looks_like_principal_id(sender_name)):
                    # 先查缓存
                    if sender_id in user_id_to_name:
                        sender_name = user_id_to_name[sender_id]
                    else:
                        # 缓存没有就查API
                        user_info = _get_user_info(client, sender_id)
                        sender_name = str(user_info.get("name") or "").strip()
                        user_id_to_name[sender_id] = sender_name

                if sender_id and sender_name and not _looks_like_principal_id(sender_name):
                    user_id_to_name[sender_id] = sender_name

                text = _normalize_message_text(msg, user_id_to_name)
                if _is_noise_message_text(text):
                    continue
                if not _is_within_recent_days(str(msg.get("create_time") or ""), recent_days):
                    continue
                normalized_message = {
                    "message_id": msg.get("message_id", ""),
                    "chat_id": msg.get("chat_id", chat_id),
                    "sender_name": sender_name,
                    "sender_id": sender_id,
                    "mentions_target_user": _message_mentions_target_user(msg, target_aliases),
                    "content": text,
                    "text": text,
                    "create_time": msg.get("create_time", ""),
                    "sender": sender,
                    "message_url": _resolve_online_message_url(
                        message=msg,
                    ),
                }
                all_messages.append(normalized_message)
                if sender_id and resolved_target and sender_id == resolved_target:
                    continue
                messages.append(normalized_message)
        except (FeishuAPIError, MissingFeishuConfigError):
            # 跳过无权限或配置错误的群，不影响整体流程
            continue

    messages = apply_interest_filter_annotations(messages)
    all_messages = apply_interest_filter_annotations(all_messages)

    drive_docs: list[dict[str, Any]] = []
    behavior_drive_docs: list[dict[str, Any]] = []
    drive_error = ""
    try:
        drive_docs = list_recent_drive_docs(client, max_items=max_drive_docs, recent_days=recent_days)
        behavior_drive_docs = list_drive_docs(client, max_items=max(max_drive_docs, max_behavior_docs), recent_days=None)
    except FeishuAPIError as exc:
        drive_error = str(exc)

    linked_docs, link_access_records = extract_docs_and_access_from_messages(all_messages, resolved_target)
    behavior_documents = _merge_documents(behavior_drive_docs, linked_docs)
    direct_access_records = _collect_direct_doc_access_records(
        client,
        documents=behavior_documents,
        target_user_id=resolved_target,
        recent_days=recent_days,
    )

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
            else:
                # 查询失败 → 用token前8位拼标题，不暴露完整ID
                doc["title"] = f"文档（{doc_token[:8]}...）"
        except Exception:
            # 查询失败 → 用token前8位拼标题，不暴露完整ID
            doc["title"] = f"文档（{doc_token[:8]}...）"

    documents = _merge_documents(drive_docs, linked_docs)
    access_records = _merge_access_records([*link_access_records, *direct_access_records])
    knowledge_items = build_knowledge_items(messages=messages, documents=documents, max_items=max_knowledge)

    return {
        "documents": documents,
        "behavior_documents": behavior_documents,
        "access_records": access_records,
        "messages": messages,
        "knowledge_items": knowledge_items,
        "resolved_target_user_id": resolved_target,
        "meta": {
            "chat_count": len(chat_records),
            "message_count": len(messages),
            "document_count": len(documents),
            "behavior_document_count": len(behavior_documents),
            "recent_days": recent_days,
            "drive_error": drive_error,
        },
    }


def list_recent_drive_docs(client: FeishuClient, max_items: int = 50, recent_days: int = 7) -> list[dict[str, Any]]:
    return list_drive_docs(client, max_items=max_items, recent_days=recent_days)


def list_drive_docs(
    client: FeishuClient,
    *,
    max_items: int = 50,
    recent_days: int | None = None,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    page_token = ""
    while len(docs) < max_items:
        page = client.list_drive_files(page_size=min(50, max_items - len(docs)), page_token=page_token)
        items = page.get("items", []) if isinstance(page, dict) else []
        for item in items:
            updated_at = str(item.get("modified_time") or item.get("edit_time") or "")
            if recent_days is not None and not _is_within_recent_days(updated_at, recent_days):
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
                    "title": key,  # 默认用doc_id占位，查询到真实标题再替换
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
            # Chat messages only prove the user shared a doc link in chat.
            # They should not be treated as direct document views.
            "action": "share",
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


def _collect_direct_doc_access_records(
    client: FeishuClient,
    *,
    documents: list[dict[str, Any]],
    target_user_id: str,
    recent_days: int,
) -> list[dict[str, Any]]:
    if not target_user_id:
        return []
    if not hasattr(client, "list_file_view_records") and not hasattr(client, "list_file_comments"):
        return []

    records: list[dict[str, Any]] = []
    for doc in documents:
        doc_id = str(doc.get("doc_id") or "").strip()
        if not doc_id:
            continue
        file_type = _infer_drive_file_type(doc)
        if file_type == "wiki":
            continue

        records.extend(
            _collect_direct_view_records(
                client,
                doc_id=doc_id,
                file_type=file_type,
                target_user_id=target_user_id,
                recent_days=recent_days,
            )
        )
        records.extend(
            _collect_direct_comment_records(
                client,
                doc_id=doc_id,
                file_type=file_type,
                target_user_id=target_user_id,
            )
        )
    return records


def _collect_direct_view_records(
    client: FeishuClient,
    *,
    doc_id: str,
    file_type: str,
    target_user_id: str,
    recent_days: int,
) -> list[dict[str, Any]]:
    if not hasattr(client, "list_file_view_records"):
        return []
    count = 0
    page_token = ""
    while True:
        try:
            page = client.list_file_view_records(
                doc_id,
                file_type=file_type,
                page_size=50,
                page_token=page_token,
            )
        except (AttributeError, FeishuAPIError, MissingFeishuConfigError):
            return []
        items = page.get("items", []) if isinstance(page, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            actor_ids = _extract_actor_ids(item)
            if target_user_id not in actor_ids:
                continue
            record_time = str(
                item.get("view_time")
                or item.get("create_time")
                or item.get("timestamp")
                or ""
            )
            if record_time and not _is_within_recent_days(record_time, recent_days):
                continue
            count += 1
        page_token = str(page.get("page_token", "") or "")
        if not page.get("has_more") or not page_token:
            break
    if count <= 0:
        return []
    return [{"doc_id": doc_id, "user_id": target_user_id, "action": "view", "count": count}]


def _collect_direct_comment_records(
    client: FeishuClient,
    *,
    doc_id: str,
    file_type: str,
    target_user_id: str,
) -> list[dict[str, Any]]:
    if file_type not in {"doc", "docx", "sheet"}:
        return []
    if not hasattr(client, "list_file_comments"):
        return []
    count = 0
    page_token = ""
    while True:
        try:
            page = client.list_file_comments(
                doc_id,
                file_type=file_type,
                page_size=50,
                page_token=page_token,
                is_solved=None,
            )
        except (AttributeError, FeishuAPIError, MissingFeishuConfigError):
            return []
        items = page.get("items", []) if isinstance(page, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            count += _count_user_comment_replies(item, target_user_id)
        page_token = str(page.get("page_token", "") or "")
        if not page.get("has_more") or not page_token:
            break
    if count <= 0:
        return []
    return [{"doc_id": doc_id, "user_id": target_user_id, "action": "comment", "count": count}]


def _count_user_comment_replies(comment_item: dict[str, Any], target_user_id: str) -> int:
    reply_list = comment_item.get("reply_list")
    if isinstance(reply_list, dict):
        replies = reply_list.get("replies", [])
        if isinstance(replies, list):
            return sum(
                1
                for reply in replies
                if isinstance(reply, dict) and target_user_id in _extract_actor_ids(reply)
            )
    return 1 if target_user_id in _extract_actor_ids(comment_item) else 0


def _merge_access_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or item.get("document_id") or item.get("token") or "").strip()
        action = str(item.get("action") or item.get("event") or "").strip().lower()
        user_id = str(item.get("user_id") or item.get("operator_id") or "").strip()
        if not doc_id or not action:
            continue
        key = (doc_id, user_id, action)
        if key not in merged:
            merged[key] = {
                "doc_id": doc_id,
                "user_id": user_id,
                "action": action,
                "count": 0,
            }
        merged[key]["count"] += int(item.get("count") or 1)
    return list(merged.values())


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


def _infer_drive_file_type(doc: dict[str, Any]) -> str:
    url = str(doc.get("url") or "").lower()
    if "/docx/" in url:
        return "docx"
    if "/docs/" in url or "/doc/" in url:
        return "doc"
    if "/sheet/" in url or "/sheets/" in url:
        return "sheet"
    if "/base/" in url:
        return "base"
    if "/slides/" in url:
        return "slides"
    if "/wiki/" in url:
        return "wiki"
    if "/file/" in url:
        return "file"
    return "docx"


def _extract_actor_ids(item: dict[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for key in ("open_id", "user_id", "operator_id", "create_user_id", "creator_id", "commenter_id", "viewer_id"):
        value = item.get(key)
        if value:
            candidates.add(str(value))
    for key in ("user", "operator", "creator", "commenter", "viewer"):
        nested = item.get(key)
        if isinstance(nested, dict):
            for nested_key in ("open_id", "user_id", "id"):
                value = nested.get(nested_key)
                if value:
                    candidates.add(str(value))
    return {value for value in candidates if value}


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s\]>\)\"]+", text)


def _looks_like_doc_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    return any(hint in host for hint in _DOC_HOST_HINTS) and any(hint in path for hint in _DOC_PATH_HINTS)


def _doc_key_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or ""
    match = re.search(r"/(?:docx|wiki|base|sheet|slides|file)/([^/?#]+)", path, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    parts = [part for part in path.split("/") if part]
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


def _normalize_message_text(
    message: dict[str, Any] | str,
    user_id_to_name: dict[str, str] | None = None,
) -> str:
    raw_text = message.get("content", "") if isinstance(message, dict) else message
    mentions = message.get("mentions", []) if isinstance(message, dict) else []
    normalized = re.sub(r"\s+", " ", str(raw_text or "").strip())
    if user_id_to_name is None:
        user_id_to_name = {}

    mention_key_to_name: dict[str, str] = {}
    for mention in mentions if isinstance(mentions, list) else []:
        if not isinstance(mention, dict):
            continue
        mention_name = str(mention.get("name") or "").strip()
        if not mention_name:
            continue
        mention_key = str(mention.get("key") or "").strip()
        mention_id = str(mention.get("id") or "").strip()
        if mention_key:
            mention_key_to_name[mention_key] = mention_name
            if mention_key.startswith("@"):
                mention_key_to_name[mention_key.removeprefix("@")] = mention_name
        if mention_id and mention_id.startswith("ou_"):
            user_id_to_name.setdefault(mention_id, mention_name)

    for mention_key in sorted(mention_key_to_name, key=len, reverse=True):
        replacement = mention_key_to_name[mention_key]
        if mention_key.startswith("@"):
            normalized = normalized.replace(mention_key, f"@{replacement}")
        else:
            normalized = normalized.replace(mention_key, replacement)

    # 替换已知的用户ID，不再为正文里的占位 mention key 额外调通讯录
    for user_id, name in user_id_to_name.items():
        normalized = normalized.replace(f"@{user_id}", f"@{name}")
        normalized = normalized.replace(user_id, name)

    return normalized


def _build_target_aliases(
    client: FeishuClient,
    resolved_target: str,
    user_id_to_name: dict[str, str],
) -> set[str]:
    aliases: set[str] = set()
    normalized_target = str(resolved_target or "").strip()
    if not normalized_target:
        return aliases
    aliases.add(normalized_target)
    if normalized_target.startswith("ou_"):
        aliases.add(normalized_target.removeprefix("ou_"))
        user_info = _get_user_info(client, normalized_target)
        display_name = str(user_info.get("name") or "").strip()
        if display_name:
            aliases.add(display_name)
            user_id_to_name[normalized_target] = display_name
    return {alias for alias in aliases if alias}


def _message_mentions_target_user(message: Any, target_aliases: set[str]) -> bool:
    if not target_aliases:
        return False
    raw = ""
    mentions: list[dict[str, Any]] = []
    if isinstance(message, dict):
        raw = str(message.get("content") or message.get("raw_content") or "")
        raw_mentions = message.get("mentions", [])
        if isinstance(raw_mentions, list):
            mentions = [item for item in raw_mentions if isinstance(item, dict)]
    else:
        raw = str(message or "")
    lowered = raw.lower()
    for alias in target_aliases:
        normalized_alias = str(alias).strip()
        if not normalized_alias:
            continue
        lowered_alias = normalized_alias.lower()
        for mention in mentions:
            mention_id = str(mention.get("id") or "").strip().lower()
            mention_name = str(mention.get("name") or "").strip().lower()
            mention_key = str(mention.get("key") or "").strip().lower()
            if lowered_alias in {mention_id, mention_name, mention_key, mention_key.removeprefix("@")}:
                return True
        if f"@{normalized_alias}".lower() in lowered:
            return True
        if f"_user_{normalized_alias}".lower() in lowered:
            return True
        if normalized_alias.startswith("ou_") and normalized_alias.lower() in lowered:
            return True
    return False


def _should_skip_message(message: dict[str, Any]) -> bool:
    msg_type = str(message.get("msg_type") or "").strip().lower()
    if msg_type == "system":
        return True
    return False


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


def _get_user_info(client: FeishuClient, user_id: str) -> dict[str, str]:
    """调用通讯录API获取用户信息"""
    if not user_id or not user_id.startswith("ou_"):
        return {}
    if getattr(client, "_contact_user_lookup_disabled", False):
        return {}
    try:
        path = f"/open-apis/contact/v3/users/{user_id}?user_id_type=open_id"
        data = client.request("GET", path)
    except Exception:
        try:
            data = client.request("GET", path, access_token=client.get_tenant_access_token())
        except Exception:
            setattr(client, "_contact_user_lookup_disabled", True)
            return {}
    try:
        user = data.get("data", {}).get("user", {})
        return {
            "name": user.get("name", ""),
            "avatar": user.get("avatar", {}).get("avatar_72", ""),
        }
    except Exception:
        return {}


def _looks_like_principal_id(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    return bool(re.fullmatch(r"(ou|cli)_[a-z0-9]+", normalized))


def _get_doc_meta(client: FeishuClient, doc_token: str, doc_type: str = "docx") -> dict[str, Any]:
    """独立的文档元数据查询功能，不依赖FeishuClient的内置方法"""
    def _request_with_fallback(path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return client.request("GET", path, params=params)
        except (FeishuAPIError, MissingFeishuConfigError):
            return client.request("GET", path, params=params, access_token=client.get_tenant_access_token())

    try:
        if doc_type == "wiki":
            data = _request_with_fallback(
                "/open-apis/wiki/v2/spaces/get_node",
                params={"token": doc_token},
            )
            node = data.get("data", {}).get("node", {})
            return {
                "title": str(node.get("title") or node.get("name") or ""),
                "url": str(node.get("url") or ""),
                "updated_at": str(node.get("obj_edit_time") or node.get("updated_time") or ""),
            }

        drive_data = _request_with_fallback(f"/open-apis/drive/v1/files/{doc_token}")
        body = drive_data.get("data", drive_data)
        title = str(body.get("name") or body.get("title") or "").strip()
        if title:
            return {
                "title": title,
                "url": str(body.get("url") or ""),
                "updated_at": str(body.get("modified_time") or body.get("edit_time") or ""),
            }
    except Exception:
        pass

    try:
        type_path_map = {
            "docx": f"/open-apis/docx/v1/documents/{doc_token}",
            "base": f"/open-apis/bitable/v1/apps/{doc_token}",
            "sheet": f"/open-apis/sheets/v3/spreadsheets/{doc_token}",
            "slides": f"/open-apis/slides/v1/presentations/{doc_token}",
        }
        path = type_path_map.get(doc_type)
        if not path:
            return {}
        data = _request_with_fallback(path)
        body = data.get("data", data)
        return {
            "title": str(body.get("name") or body.get("title") or ""),
            "url": str(body.get("url") or ""),
            "updated_at": str(body.get("modified_time") or body.get("edit_time") or ""),
        }
    except Exception:
        return {}


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
