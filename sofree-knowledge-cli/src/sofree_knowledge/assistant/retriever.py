from __future__ import annotations

from typing import Any

from .models import (
    DocumentCandidate,
    DocumentRetrievalDebug,
    DualTowerConfig,
    RelatedKnowledge,
    RelatedMessage,
)
from .training import load_dual_tower_model, score_dual_tower_with_model
from .two_tower import build_dual_tower_debug_payload, score_dual_tower_texts


def rank_related_messages(
    doc: dict[str, str],
    doc_keywords: set[str],
    messages: list[dict[str, str]],
    max_related: int,
) -> list[dict[str, Any]]:
    from .. import assistant_brief as legacy

    return legacy._rank_related_messages(doc, doc_keywords, messages, max_related)


def rank_related_knowledge(
    doc_keywords: set[str],
    knowledge: list[dict[str, str]],
    max_related: int,
) -> list[dict[str, Any]]:
    from .. import assistant_brief as legacy

    return legacy._rank_related_knowledge(doc_keywords, knowledge, max_related)


def build_interest_digest(messages: list[dict[str, Any]], interests: list[str], max_items: int) -> dict[str, Any]:
    from .. import assistant_brief as legacy

    return legacy._build_interest_digest(messages, interests, max_items)


def build_document_candidate(
    *,
    doc: dict[str, str],
    doc_type: str,
    business: str,
    title_keywords: set[str],
    messages: list[dict[str, Any]],
    knowledge: list[dict[str, Any]],
    profile: dict[str, Any],
    max_related: int,
    dual_tower_config: DualTowerConfig | None = None,
) -> DocumentCandidate:
    use_dual_tower = bool((dual_tower_config or DualTowerConfig()).enabled)
    retrieval_strategy = "dual_tower_bootstrap" if use_dual_tower else "openclaw_fallback"
    candidate_messages = messages if use_dual_tower else _filter_messages_by_openclaw(messages)
    related_messages_raw = rank_related_messages(doc, title_keywords, candidate_messages, max_related=max_related)
    related_knowledge_raw = rank_related_knowledge(title_keywords, knowledge, max_related=max_related)
    dual_tower_debug = build_dual_tower_debug_payload(
        profile,
        doc,
        business=business,
        doc_type=doc_type,
        config=dual_tower_config,
    )
    dual_tower_score = (
        score_dual_tower_texts(
            str(dual_tower_debug.get("user_tower_text", "")),
            str(dual_tower_debug.get("content_tower_text", "")),
        )
        if use_dual_tower
        else 0.0
    )
    model_file = str(getattr(dual_tower_config, "model_file", "") or "")
    trained_dual_tower_score = 0.0
    model_applied = False
    if use_dual_tower and model_file:
        model = load_dual_tower_model(model_file)
        trained_dual_tower_score = score_dual_tower_with_model(
            str(dual_tower_debug.get("user_tower_text", "")),
            str(dual_tower_debug.get("content_tower_text", "")),
            model,
        )
        model_applied = True
        retrieval_strategy = "dual_tower_trained"
    lexical_overlap = max((int(item.get("score", 0)) // 20 for item in related_messages_raw), default=0)
    return DocumentCandidate(
        doc_id=doc.get("doc_id", ""),
        doc_type=doc_type,
        business=business,
        title_keywords=set(title_keywords),
        related_messages=[
            RelatedMessage(
                message_id=str(item.get("message_id", "")),
                chat_id=str(item.get("chat_id", "")),
                summary=str(item.get("summary", "")),
                score=int(item.get("score", 0)),
                create_time=str(item.get("create_time", "")),
            )
            for item in related_messages_raw
        ],
        related_knowledge=[
            RelatedKnowledge(
                id=str(item.get("id", "")),
                title=str(item.get("title", "")),
                summary=str(item.get("summary", "")),
                score=int(item.get("score", 0)),
            )
            for item in related_knowledge_raw
        ],
        retrieval_debug=DocumentRetrievalDebug(
            strategy=retrieval_strategy,
            dual_tower_score=dual_tower_score,
            trained_dual_tower_score=trained_dual_tower_score,
            lexical_overlap=lexical_overlap,
            related_message_count=len(related_messages_raw),
            related_knowledge_count=len(related_knowledge_raw),
            user_tower_text=str(dual_tower_debug.get("user_tower_text", "")) if use_dual_tower else "",
            content_tower_text=str(dual_tower_debug.get("content_tower_text", "")) if use_dual_tower else "",
            dual_tower_ready=use_dual_tower,
            model_applied=model_applied,
            model_file=model_file,
        ),
    )


def _filter_messages_by_openclaw(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .. import assistant_brief as legacy

    accepted: list[dict[str, Any]] = []
    for message in messages:
        signal_text = str(message.get("summary_text") or message.get("text") or "")
        lowered = signal_text.lower()
        signal_hits = sum(1 for keyword in legacy.INTEREST_SIGNAL_KEYWORDS if keyword in lowered)
        urgency_hit = any(keyword in lowered for keyword in legacy.URGENCY_KEYWORDS)
        hit_term_count = max(1, signal_hits) if signal_hits > 0 else 0
        mention_signal = legacy._mention_signal_text(message, signal_text)
        screen = legacy._openclaw_interest_screen(
            message=message,
            signal_hits=signal_hits,
            urgency_hit=urgency_hit,
            hit_term_count=hit_term_count,
            mention_signal=mention_signal,
        )
        if screen.get("accepted"):
            accepted.append(message)
    return accepted or messages
