from __future__ import annotations

import json

import llm.client as llm_client

from ..prompt import get_prompt
from ..shared.models import LiftedCard, RouteDecision


def _route_by_rule(card: LiftedCard) -> tuple[str, list[str]]:
    reason_codes: list[str] = []
    target = "observe"
    signals = card.decision_signals or {}
    actionability_score = float(signals.get("actionability_score", 0.0) or 0.0)
    novelty_score = float(signals.get("novelty_score", 0.0) or 0.0)
    impact_score = float(signals.get("impact_score", 0.0) or 0.0)
    confidence = float(card.confidence or 0.0)
    has_question = float(signals.get("has_question", 0.0) or 0.0) >= 0.5
    has_action_hint = float(signals.get("has_action_hint", 0.0) or 0.0) >= 0.5
    role = str(card.message_role or "").strip().lower()
    missing_fields = [str(x).strip().lower() for x in (card.missing_fields or []) if str(x).strip()]
    is_task_like = any(field in {"owner", "time", "location", "deadline"} for field in missing_fields)

    # 1) Respect lifted target first.
    if card.suggested_target == "knowledge":
        target = "knowledge"
        reason_codes.append("suggested-knowledge")
    elif card.suggested_target == "task":
        target = "task"
        reason_codes.append("suggested-task")
    # 2) Context role and action semantics.
    elif role in {"followup", "update", "confirm"} and (actionability_score >= 0.5 or has_action_hint):
        target = "task"
        reason_codes.extend(["context-followup-task", "action-signal"])
    elif role == "question" and (has_question or actionability_score >= 0.5):
        target = "task"
        reason_codes.extend(["context-question-task", "question-signal"])
    # 3) Tags and explicit task incompleteness.
    elif "actionable-signal" in card.tags or is_task_like:
        target = "task"
        reason_codes.append("task-semantic-signal")
    elif "novel-content" in card.tags or novelty_score >= 0.6:
        target = "knowledge"
        reason_codes.append("knowledge-semantic-signal")
    else:
        # Soft score reference (not hard threshold gating): use score tendency as tie-breaker.
        task_score = actionability_score + (0.15 if has_action_hint else 0.0) + (0.1 if has_question else 0.0) + (0.1 if confidence >= 0.7 else 0.0)
        knowledge_score = novelty_score + (impact_score * 0.4) + (0.1 if confidence >= 0.7 else 0.0)
        if task_score >= 0.85 and task_score >= knowledge_score:
            target = "task"
            reason_codes.append("score-lean-task")
        elif knowledge_score >= 0.85 and knowledge_score > task_score:
            target = "knowledge"
            reason_codes.append("score-lean-knowledge")
        else:
            reason_codes.append("fallback-observe")
    return target, reason_codes


def _route_by_llm(
    card: LiftedCard,
) -> tuple[str, list[str]] | None:
    # 路由判断任务：三选一输出，非常简单，使用最快参数
    config = llm_client.LLMConfig.from_env(
        max_tokens=96,
        temperature=0.0,
        top_p=0.1,
        extra_body={"thinking": {"type": "disabled"}},
    )
    if config.missing_fields():
        return None
    try:
        system_prompt = get_prompt("route_v2.system_prompt")
        user_prompt = get_prompt("route_v2.user_prompt").format(
            card_json=json.dumps(card.to_dict(), ensure_ascii=False),
        )
    except Exception:
        return None
    payload = llm_client.invoke_structured(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=llm_client.RouteOutput,
    )
    if payload is None:
        return None
    target = str(payload.target_pool).strip().lower()
    if target not in {"knowledge", "task", "observe"}:
        return None
    reasons = [str(x).strip() for x in payload.reason_codes if str(x).strip()]
    return target, (reasons or ["llm-route"])


def route_cards(
    cards: list[LiftedCard],
) -> list[RouteDecision]:
    decisions: list[RouteDecision] = []
    snapshot = {"routing_mode": "semantic_with_score_reference"}
    for card in cards:
        llm_decision = _route_by_llm(card)
        if llm_decision is None:
            target, reason_codes = _route_by_rule(card)
        else:
            target, reason_codes = llm_decision
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
