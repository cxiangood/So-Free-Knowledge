from __future__ import annotations

import json
import re
from typing import Any

from llm.client import LLMClient, LLMConfig

from ..prompt import get_prompt
from ..shared.models import LiftedCard, RouteDecision


def _route_by_rule(card: LiftedCard, *, knowledge_threshold: float, task_threshold: float) -> tuple[str, list[str]]:
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
    return target, reason_codes


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _route_by_llm(
    card: LiftedCard,
    *,
    knowledge_threshold: float,
    task_threshold: float,
) -> tuple[str, list[str]] | None:
    # 路由判断任务：三选一输出，非常简单，使用最快参数
    config = LLMConfig.from_env(max_tokens=64, temperature=0.0, top_p=0.1)
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
    response = LLMClient(config).build_reply(system_prompt, user_prompt)
    if response.startswith("LLM "):
        return None
    payload = _extract_json(response)
    if not payload:
        return None
    target = str(payload.get("target_pool", "")).strip().lower()
    if target not in {"knowledge", "task", "observe"}:
        return None
    reasons_raw = payload.get("reason_codes", [])
    reasons = [str(x).strip() for x in reasons_raw if str(x).strip()] if isinstance(reasons_raw, list) else []
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
