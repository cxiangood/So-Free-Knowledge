from __future__ import annotations

from typing import Any


def score_doc_urgency(doc: dict[str, str], related_messages: list[dict[str, Any]]) -> int:
    from .. import assistant_brief as legacy

    return legacy._score_doc_urgency(doc, related_messages)


def score_doc_recommend(
    doc: dict[str, str],
    stats: dict[str, int],
    related_messages: list[dict[str, Any]],
    related_knowledge: list[dict[str, Any]],
) -> int:
    from .. import assistant_brief as legacy

    return legacy._score_doc_recommend(doc, stats, related_messages, related_knowledge)
