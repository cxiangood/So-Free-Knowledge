from __future__ import annotations

import json
import logging

import llm.client as llm_client
from utils import get_config_float, get_config_int, get_config_str

from ..prompt import get_prompt
from ..shared.models import LiftedCard, RouteDecision


_ROUTE_ORDER = ("knowledge", "task", "observe")
LOGGER = logging.getLogger(__name__)


def _normalize_routes(routes: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    merged: dict[str, list[str]] = {}
    for raw_target, raw_reasons in routes:
        target = str(raw_target).strip().lower()
        if target not in _ROUTE_ORDER:
            continue
        reasons = merged.setdefault(target, [])
        for reason in raw_reasons or []:
            value = str(reason).strip()
            if value and value not in reasons:
                reasons.append(value)
    if not merged:
        merged["observe"] = ["fallback-observe"]
    return [(target, merged[target] or ["llm-route"]) for target in _ROUTE_ORDER if target in merged]


def _route_by_rule(card: LiftedCard) -> list[tuple[str, list[str]]]:
    routes: list[tuple[str, list[str]]] = []
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

    if card.suggested_target == "knowledge":
        routes.append(("knowledge", ["suggested-knowledge"]))
    elif card.suggested_target == "task":
        routes.append(("task", ["suggested-task"]))
    elif card.suggested_target == "observe":
        routes.append(("observe", ["suggested-observe"]))

    if role in {"followup", "update", "confirm", "cancel"} and (actionability_score >= 0.5 or has_action_hint):
        routes.append(("task", ["context-followup-task", "action-signal"]))
    if role == "question" and (has_question or actionability_score >= 0.5):
        routes.append(("task", ["context-question-task", "question-signal"]))
    if "actionable-signal" in card.tags or is_task_like:
        routes.append(("task", ["task-semantic-signal"]))
    if "novel-content" in card.tags or novelty_score >= 0.6:
        routes.append(("knowledge", ["knowledge-semantic-signal"]))

    if not routes:
        # Soft score reference (not hard threshold gating): use score tendency as tie-breaker.
        task_score = actionability_score + (0.15 if has_action_hint else 0.0) + (0.1 if has_question else 0.0) + (0.1 if confidence >= 0.7 else 0.0)
        knowledge_score = novelty_score + (impact_score * 0.4) + (0.1 if confidence >= 0.7 else 0.0)
        if task_score >= 0.85 and task_score >= knowledge_score:
            routes.append(("task", ["score-lean-task"]))
        elif knowledge_score >= 0.85 and knowledge_score > task_score:
            routes.append(("knowledge", ["score-lean-knowledge"]))
        else:
            routes.append(("observe", ["fallback-observe"]))
    if role == "chitchat" or not routes:
        routes.append(("observe", ["weak-or-unclear-signal", "fallback-observe"]))
    return _normalize_routes(routes)


def _route_by_llm(
    card: LiftedCard,
) -> list[tuple[str, list[str]]] | None:
    # 路由判断任务：输出 1~3 个目标池，保持低温度保证稳定。
    thinking_type = get_config_str("insight.llm.route.thinking_type", "disabled").strip()
    config = llm_client.LLMConfig.from_env(
        max_tokens=get_config_int("insight.llm.route.max_tokens", 256),
        temperature=get_config_float("insight.llm.route.temperature", 0.0),
        top_p=get_config_float("insight.llm.route.top_p", 0.1),
        extra_body={"thinking": {"type": thinking_type}} if thinking_type else None,
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
    routes = _normalize_routes([(item.target_pool, item.reason_codes) for item in payload.routes])
    if not routes:
        return None
    return routes


def route_cards(
    cards: list[LiftedCard],
    *,
    message_id: str = "",
    chat_id: str = "",
) -> list[RouteDecision]:
    decisions: list[RouteDecision] = []
    snapshot = {"routing_mode": "semantic_with_score_reference"}
    for card in cards:
        routes = _route_by_llm(card)
        if routes is None:
            LOGGER.warning(
                "fallback module=route reason=llm_failed strategy=rule_routing message_id=%s chat_id=%s card_id=%s",
                message_id or "-",
                chat_id or "-",
                card.card_id,
            )
            routes = _route_by_rule(card)
        for target, reason_codes in routes:
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
