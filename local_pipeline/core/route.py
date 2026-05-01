from __future__ import annotations

import json

import llm.client as llm_client

from ..prompt import get_prompt
from ..shared.models import LiftedCard, RouteDecision


def _route_by_rule(card: LiftedCard, *, knowledge_threshold: float, task_threshold: float) -> tuple[str, list[str]]:
    reason_codes: list[str] = []
    target = "observe"
    actionability_score = float((card.decision_signals or {}).get("actionability_score", 0.0))
    role = str(card.message_role or "").strip().lower()
    if card.suggested_target == "knowledge" and card.confidence >= knowledge_threshold:
        target = "knowledge"
        reason_codes.extend(["suggested-knowledge", "confidence-pass"])
    elif card.suggested_target == "task" and card.confidence >= task_threshold:
        target = "task"
        reason_codes.extend(["suggested-task", "confidence-pass"])
    elif role in {"followup", "update", "confirm"} and card.confidence >= task_threshold and actionability_score >= 0.50:
        target = "task"
        reason_codes.extend(["context-followup-task", "actionability-pass"])
    elif card.confidence >= task_threshold and ("actionable-signal" in card.tags):
        target = "task"
        reason_codes.extend(["rule-task", "actionable-tag"])
    elif card.confidence >= knowledge_threshold and ("novel-content" in card.tags):
        target = "knowledge"
        reason_codes.extend(["rule-knowledge", "novel-tag"])
    else:
        reason_codes.append("fallback-observe")
    return target, reason_codes


def _route_by_llm(
    card: LiftedCard,
    *,
    knowledge_threshold: float,
    task_threshold: float,
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
        system_prompt = get_prompt("route.system_prompt")
        user_prompt = get_prompt("route.user_prompt").format(
            card_json=json.dumps(card.to_dict(), ensure_ascii=False),
            knowledge_threshold=knowledge_threshold,
            task_threshold=task_threshold,
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
        llm_decision = _route_by_llm(card, knowledge_threshold=knowledge_threshold, task_threshold=task_threshold)
        if llm_decision is None:
            target, reason_codes = _route_by_rule(card, knowledge_threshold=knowledge_threshold, task_threshold=task_threshold)
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
