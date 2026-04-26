"""Personal assistant brief builder from docs/access/messages/knowledge."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


URGENCY_KEYWORDS = (
    "紧急",
    "马上",
    "尽快",
    "asap",
    "今天",
    "今晚",
    "截止",
    "ddl",
    "线上",
    "故障",
    "报错",
    "异常",
    "阻塞",
    "p0",
    "p1",
)


def build_personal_brief(
    documents: list[dict[str, Any]],
    access_records: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    knowledge_items: list[dict[str, Any]] | None = None,
    target_user_id: str = "",
    max_docs: int = 10,
    max_related: int = 5,
) -> dict[str, Any]:
    docs = [_normalize_doc(item) for item in documents]
    records = [item for item in (access_records or []) if isinstance(item, dict)]
    msgs = [_normalize_message(item) for item in (messages or []) if isinstance(item, dict)]
    knowledge = [_normalize_knowledge(item) for item in (knowledge_items or []) if isinstance(item, dict)]

    doc_stats = _build_doc_stats(records, target_user_id=target_user_id)

    ranked_docs: list[dict[str, Any]] = []
    for doc in docs:
        title_keywords = _extract_keywords(" ".join([doc["title"], doc["summary"]]))
        related_messages = _rank_related_messages(doc, title_keywords, msgs, max_related=max_related)
        related_knowledge = _rank_related_knowledge(title_keywords, knowledge, max_related=max_related)

        urgency_score = _score_doc_urgency(doc=doc, related_messages=related_messages)
        recommend_score = _score_doc_recommend(
            doc=doc,
            stats=doc_stats.get(doc["doc_id"], {}),
            related_messages=related_messages,
            related_knowledge=related_knowledge,
        )
        ranked_docs.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "url": doc["url"],
                "summary": doc["summary"],
                "recommend_score": recommend_score,
                "urgency_score": urgency_score,
                "priority": _priority_label(urgency_score, recommend_score),
                "reasons": _build_reasons(doc, doc_stats.get(doc["doc_id"], {}), related_messages),
                "related_messages": related_messages,
                "related_knowledge": related_knowledge,
            }
        )

    ranked_docs.sort(
        key=lambda item: (int(item["urgency_score"]), int(item["recommend_score"])),
        reverse=True,
    )
    ranked_docs = ranked_docs[:max_docs]

    summary = {
        "doc_count": len(ranked_docs),
        "critical_count": sum(1 for item in ranked_docs if item["priority"] == "critical"),
        "high_count": sum(1 for item in ranked_docs if item["priority"] == "high"),
        "normal_count": sum(1 for item in ranked_docs if item["priority"] == "normal"),
    }

    doc_markdown = _to_markdown(ranked_docs)
    card = _to_card(ranked_docs)
    return {
        "summary": summary,
        "documents": ranked_docs,
        "doc_markdown": doc_markdown,
        "card": card,
    }


def _normalize_doc(item: dict[str, Any]) -> dict[str, str]:
    return {
        "doc_id": str(item.get("doc_id") or item.get("id") or item.get("token") or ""),
        "title": str(item.get("title") or item.get("name") or ""),
        "summary": str(item.get("summary") or item.get("description") or ""),
        "url": str(item.get("url") or item.get("link") or ""),
        "updated_at": str(item.get("updated_at") or item.get("update_time") or item.get("timestamp") or ""),
    }


def _normalize_message(item: dict[str, Any]) -> dict[str, str]:
    text = item.get("text")
    if text is None:
        text = item.get("content", "")
    return {
        "message_id": str(item.get("message_id") or item.get("id") or ""),
        "chat_id": str(item.get("chat_id") or ""),
        "text": str(text or ""),
        "create_time": str(item.get("create_time") or item.get("timestamp") or ""),
    }


def _normalize_knowledge(item: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(item.get("id") or item.get("knowledge_id") or ""),
        "title": str(item.get("title") or item.get("keyword") or ""),
        "content": str(item.get("content") or item.get("value") or item.get("summary") or ""),
    }


def _build_doc_stats(records: list[dict[str, Any]], target_user_id: str) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"views": 0, "edits": 0, "comments": 0})
    normalized_target = str(target_user_id or "").strip()
    for record in records:
        doc_id = str(record.get("doc_id") or record.get("document_id") or record.get("token") or "").strip()
        if not doc_id:
            continue
        if normalized_target:
            user_id = str(record.get("user_id") or record.get("operator_id") or "").strip()
            if user_id and user_id != normalized_target:
                continue
        action = str(record.get("action") or record.get("event") or "").lower()
        count = int(record.get("count") or 1)
        if "edit" in action:
            stats[doc_id]["edits"] += count
        elif "comment" in action:
            stats[doc_id]["comments"] += count
        else:
            stats[doc_id]["views"] += count
    return stats


def _extract_keywords(text: str) -> set[str]:
    normalized = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    result: set[str] = {token for token in tokens if token.strip()}
    # Add Chinese 2-char grams to improve partial semantic matching.
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            for idx in range(0, len(token) - 1):
                result.add(token[idx : idx + 2])
    return result


def _rank_related_messages(
    doc: dict[str, str],
    doc_keywords: set[str],
    messages: list[dict[str, str]],
    max_related: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        text = message["text"]
        if not text:
            continue
        message_keywords = _extract_keywords(text)
        overlap = len(doc_keywords & message_keywords)
        if overlap <= 0:
            continue
        urgency_hit = any(keyword in text.lower() for keyword in URGENCY_KEYWORDS)
        score = overlap * 20 + (30 if urgency_hit else 0)
        items.append(
            {
                "message_id": message["message_id"],
                "chat_id": message["chat_id"],
                "summary": _trim(text, 100),
                "score": score,
                "create_time": message["create_time"],
            }
        )
    items.sort(key=lambda item: (int(item["score"]), str(item["create_time"])), reverse=True)
    return items[:max_related]


def _rank_related_knowledge(
    doc_keywords: set[str],
    knowledge: list[dict[str, str]],
    max_related: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in knowledge:
        text = " ".join([entry["title"], entry["content"]])
        keywords = _extract_keywords(text)
        overlap = len(doc_keywords & keywords)
        if overlap <= 0:
            continue
        score = overlap * 25
        items.append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "summary": _trim(entry["content"], 80),
                "score": score,
            }
        )
    items.sort(key=lambda item: int(item["score"]), reverse=True)
    return items[:max_related]


def _score_doc_urgency(doc: dict[str, str], related_messages: list[dict[str, Any]]) -> int:
    score = 0
    text = f"{doc['title']} {doc['summary']}".lower()
    if any(keyword in text for keyword in URGENCY_KEYWORDS):
        score += 40
    score += min(40, sum(1 for item in related_messages if int(item.get("score", 0)) >= 40) * 10)
    score += min(
        20,
        sum(
            1
            for item in related_messages
            if any(keyword in str(item.get("summary", "")).lower() for keyword in URGENCY_KEYWORDS)
        )
        * 10,
    )
    return min(100, score)


def _score_doc_recommend(
    doc: dict[str, str],
    stats: dict[str, int],
    related_messages: list[dict[str, Any]],
    related_knowledge: list[dict[str, Any]],
) -> int:
    score = 20
    score += min(25, int(stats.get("views", 0)) * 3)
    score += min(20, int(stats.get("edits", 0)) * 8)
    score += min(15, int(stats.get("comments", 0)) * 5)
    score += min(10, len(related_messages) * 2)
    score += min(10, len(related_knowledge) * 3)
    if _is_recent(doc.get("updated_at", "")):
        score += 10
    return min(100, score)


def _priority_label(urgency_score: int, recommend_score: int) -> str:
    if urgency_score >= 70:
        return "critical"
    if urgency_score >= 45 or recommend_score >= 75:
        return "high"
    return "normal"


def _build_reasons(doc: dict[str, str], stats: dict[str, int], related_messages: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if int(stats.get("edits", 0)) > 0:
        reasons.append(f"近期编辑活跃（{stats.get('edits', 0)} 次）")
    if int(stats.get("views", 0)) > 0:
        reasons.append(f"近期访问较多（{stats.get('views', 0)} 次）")
    if related_messages:
        reasons.append(f"关联群聊消息 {len(related_messages)} 条")
    if not reasons:
        reasons.append("基础推荐：与近期语义上下文有关联")
    return reasons


def _is_recent(raw: str) -> bool:
    value = str(raw or "").strip()
    if not value:
        return False
    if value.isdigit():
        timestamp = int(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp // 1000
        now_ts = int(datetime.now(timezone.utc).timestamp())
        return (now_ts - timestamp) <= 7 * 24 * 3600
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).days <= 7


def _trim(text: str, max_len: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def _to_markdown(documents: list[dict[str, Any]]) -> str:
    lines = ["# 个人助理聚合简报", "", "## 文档优先级"]
    if not documents:
        lines.append("- 暂无可推荐文档")
        return "\n".join(lines)
    for index, item in enumerate(documents, start=1):
        lines.append(
            f"{index}. [{item['priority'].upper()}] {item['title']} "
            f"(紧急:{item['urgency_score']} 推荐:{item['recommend_score']})"
        )
        for reason in item["reasons"][:2]:
            lines.append(f"   - {reason}")
        if item["related_messages"]:
            lines.append(f"   - 相关消息: {item['related_messages'][0]['summary']}")
        if item["related_knowledge"]:
            lines.append(f"   - 知识建议: {item['related_knowledge'][0]['title']}")
    return "\n".join(lines)


def _to_card(documents: list[dict[str, Any]]) -> dict[str, Any]:
    content_lines: list[str] = []
    for item in documents[:5]:
        content_lines.append(
            f"- **{item['priority'].upper()}** {item['title']} "
            f"(紧急:{item['urgency_score']} 推荐:{item['recommend_score']})"
        )
    if not content_lines:
        content_lines = ["- 暂无可推荐文档"]
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "turquoise",
            "title": {"tag": "plain_text", "content": "个人助理聚合建议"},
        },
        "elements": [{"tag": "markdown", "content": "\n".join(content_lines)}],
    }
