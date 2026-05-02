from __future__ import annotations

from collections import defaultdict
from typing import Any

from .two_tower import build_document_tower_text, build_user_tower_text


def build_weak_supervision_samples(
    *,
    documents: list[dict[str, Any]],
    access_records: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    user_profile: dict[str, Any],
    target_user_id: str = "",
) -> list[dict[str, Any]]:
    docs_by_id = {
        str(item.get("doc_id") or item.get("id") or item.get("token") or "").strip(): item
        for item in documents
        if str(item.get("doc_id") or item.get("id") or item.get("token") or "").strip()
    }
    if not docs_by_id:
        return []

    recent_topics = _extract_recent_topics(messages)
    enriched_profile = dict(user_profile)
    enriched_profile.setdefault("recent_topics", recent_topics)
    user_tower_text = build_user_tower_text_with_topics(enriched_profile)

    positives: dict[str, int] = defaultdict(int)
    normalized_target = str(target_user_id or "").strip()
    for record in access_records:
        doc_id = str(record.get("doc_id") or record.get("document_id") or record.get("token") or "").strip()
        if not doc_id or doc_id not in docs_by_id:
            continue
        user_id = str(record.get("user_id") or record.get("operator_id") or "").strip()
        if normalized_target and user_id and user_id != normalized_target:
            continue
        action = str(record.get("action") or record.get("event") or "").lower()
        count = int(record.get("count") or 1)
        positives[doc_id] += _positive_strength_for_action(action, count)

    if not positives:
        return []

    negative_doc_ids = [doc_id for doc_id in docs_by_id if doc_id not in positives]
    samples: list[dict[str, Any]] = []
    for doc_id, strength in positives.items():
        doc = docs_by_id[doc_id]
        positive_text = build_document_tower_text(
            doc,
            business=str(doc.get("business") or "General"),
            doc_type=str(doc.get("doc_type") or "file"),
        )
        negatives = [
            build_document_tower_text(
                docs_by_id[negative_id],
                business=str(docs_by_id[negative_id].get("business") or "General"),
                doc_type=str(docs_by_id[negative_id].get("doc_type") or "file"),
            )
            for negative_id in negative_doc_ids[:10]
        ]
        samples.append(
            {
                "user_id": normalized_target,
                "doc_id": doc_id,
                "label": 1,
                "strength": strength,
                "user_tower_text": user_tower_text,
                "positive_doc_text": positive_text,
                "negative_doc_texts": negatives,
                "recent_topics": recent_topics,
            }
        )
    return samples


def _positive_strength_for_action(action: str, count: int) -> int:
    normalized_count = max(1, int(count or 1))
    if "edit" in action:
        return normalized_count * 3
    if "comment" in action:
        return normalized_count * 2
    if "share" in action:
        return normalized_count
    if "view" in action or "click" in action or "open" in action:
        return normalized_count * 2
    return normalized_count


def build_user_tower_text_with_topics(profile: dict[str, Any]) -> str:
    base = build_user_tower_text(profile)
    topics = [str(item).strip() for item in profile.get("recent_topics", []) if str(item).strip()]
    if not topics:
        return base
    suffix = f"recent_topics: {', '.join(topics[:10])}"
    return " | ".join(part for part in [base, suffix] if part)


def _extract_recent_topics(messages: list[dict[str, Any]]) -> list[str]:
    freq: dict[str, int] = {}
    for message in messages:
        text = str(message.get("text") or message.get("content") or "").lower()
        for token in text.split():
            normalized = token.strip(".,!?()[]{}:;\"'")
            if len(normalized) < 3:
                continue
            freq[normalized] = freq.get(normalized, 0) + 1
    return [token for token, _ in sorted(freq.items(), key=lambda item: item[1], reverse=True)[:10]]
