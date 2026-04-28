from __future__ import annotations

from ..shared.models import LiftedCard, RouteDecision


def route_cards(
    cards: list[LiftedCard],
    *,
    knowledge_threshold: float = 0.60,
    task_threshold: float = 0.50,
) -> list[RouteDecision]:
    decisions: list[RouteDecision] = []
    snapshot = {
        "knowledge_threshold": knowledge_threshold,
        "task_threshold": task_threshold,
    }
    for card in cards:
        reason_codes: list[str] = []
        target = "observe"
        if card.suggested_target == "knowledge" and card.confidence >= knowledge_threshold:
            target = "knowledge"
            reason_codes.extend(["suggested-knowledge", "confidence-pass"])
        elif card.suggested_target == "task" and card.confidence >= task_threshold:
            target = "task"
            reason_codes.extend(["suggested-task", "confidence-pass"])
        elif card.confidence >= task_threshold and ("actionable-signal" in card.tags):
            target = "task"
            reason_codes.extend(["rule-task", "actionable-tag"])
        elif card.confidence >= knowledge_threshold and ("novel-content" in card.tags):
            target = "knowledge"
            reason_codes.extend(["rule-knowledge", "novel-tag"])
        else:
            reason_codes.append("fallback-observe")
        decisions.append(
            RouteDecision(
                card_id=card.card_id,
                target_pool=target,  # type: ignore[arg-type]
                reason_codes=reason_codes,
                threshold_snapshot=dict(snapshot),
            )
        )
    return decisions

__all__ = ["route_cards"]
