from __future__ import annotations

import logging
from dataclasses import dataclass
import json

import llm.client as llm_client
from utils import get_config_float, get_config_int, get_config_str

from ..prompt import get_prompt
from ..shared.models import LiftedCard, RagHit

QUESTION_MARKERS = ("?", "？", "请问", "怎么", "如何", "吗", "是否", "能否", "为什么", "求助")
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ObserveAnswerResult:
    can_answer: bool
    answer: str = ""
    hits: list[RagHit] | None = None
    reason: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class ObserveConvertDecision:
    target_pool: str
    reason_codes: list[str]
    confidence: float = 0.0


@dataclass(slots=True)
class ObserveMergeDecision:
    action: str
    target_pool: str
    observe_ids: list[str]
    reason_codes: list[str]
    confidence: float = 0.0


def is_question_by_rule(*, summary: str, problem: str, content: str) -> bool:
    text = f"{summary}\n{problem}\n{content}".lower()
    return any(marker in text for marker in QUESTION_MARKERS)


def is_question_with_llm(*, summary: str, problem: str, content: str, message_id: str = "", chat_id: str = "") -> bool:
    rule_guess = is_question_by_rule(summary=summary, problem=problem, content=content)
    thinking_type = get_config_str("insight.llm.observe_qa.thinking_type").strip()
    config = llm_client.LLMConfig.from_env(
        max_tokens=get_config_int("insight.llm.observe_qa.max_tokens"),
        temperature=get_config_float("insight.llm.observe_qa.temperature"),
        top_p=get_config_float("insight.llm.observe_qa.top_p"),
        extra_body={"thinking": {"type": thinking_type}} if thinking_type else None,
    )
    if config.missing_fields():
        return rule_guess
    try:
        system_prompt = get_prompt("observe_qa.system_prompt")
        user_prompt = get_prompt("observe_qa.user_prompt").format(summary=summary, problem=problem, content=content)
    except Exception:
        return rule_guess
    payload = llm_client.invoke_structured(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=llm_client.ObserveQuestionOutput,
    )
    if payload is None:
        LOGGER.warning(
            "fallback module=observe_qa reason=llm_failed strategy=rule_question_detection message_id=%s chat_id=%s",
            message_id or "-",
            chat_id or "-",
        )
        return rule_guess
    if payload.is_question is not None:
        return bool(payload.is_question)
    if payload.need_reply is not None:
        return bool(payload.need_reply)
    return rule_guess


def try_answer_with_rag(query: str, hits: list[RagHit], min_hits: int = 1) -> ObserveAnswerResult:
    if not hits or len(hits) < max(1, int(min_hits or 1)):
        return ObserveAnswerResult(can_answer=False, reason="no_relevant_knowledge")
    top = hits[:3]
    thinking_type = get_config_str("insight.llm.observe_qa.thinking_type").strip()
    config = llm_client.LLMConfig.from_env(
        max_tokens=get_config_int("insight.llm.observe_qa.max_tokens"),
        temperature=get_config_float("insight.llm.observe_qa.temperature"),
        top_p=get_config_float("insight.llm.observe_qa.top_p"),
        extra_body={"thinking": {"type": thinking_type}} if thinking_type else None,
    )
    if config.missing_fields():
        return ObserveAnswerResult(can_answer=False, reason="llm_missing_config", hits=top)
    try:
        system_prompt = get_prompt("observe_answer_v1.system_prompt")
        user_prompt = get_prompt("observe_answer_v1.user_prompt").format(
            query=query,
            hits_json=json.dumps([item.to_dict() for item in top], ensure_ascii=False),
        )
    except Exception:
        return ObserveAnswerResult(can_answer=False, reason="prompt_error", hits=top)
    payload = llm_client.invoke_structured(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=llm_client.ObserveAnswerOutput,
    )
    if payload is None:
        return ObserveAnswerResult(can_answer=False, reason="llm_failed", hits=top)
    return ObserveAnswerResult(
        can_answer=bool(payload.can_answer),
        answer=str(payload.answer or "").strip(),
        hits=top,
        reason=str(payload.reason or "").strip(),
        confidence=float(payload.confidence or 0.0),
    )


def decide_non_question_target(card: LiftedCard, hits: list[RagHit]) -> ObserveConvertDecision:
    thinking_type = get_config_str("insight.llm.route.thinking_type").strip()
    config = llm_client.LLMConfig.from_env(
        max_tokens=get_config_int("insight.llm.route.max_tokens"),
        temperature=get_config_float("insight.llm.route.temperature"),
        top_p=get_config_float("insight.llm.route.top_p"),
        extra_body={"thinking": {"type": thinking_type}} if thinking_type else None,
    )
    if config.missing_fields():
        return ObserveConvertDecision(target_pool="observe", reason_codes=["llm-unavailable"])
    try:
        system_prompt = get_prompt("observe_convert_v1.system_prompt")
        user_prompt = get_prompt("observe_convert_v1.user_prompt").format(
            card_json=json.dumps(card.to_dict(), ensure_ascii=False),
            hits_json=json.dumps([item.to_dict() for item in hits[:5]], ensure_ascii=False),
        )
    except Exception:
        return ObserveConvertDecision(target_pool="observe", reason_codes=["prompt-error"])
    payload = llm_client.invoke_structured(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=llm_client.ObserveConvertOutput,
    )
    if payload is None:
        return ObserveConvertDecision(target_pool="observe", reason_codes=["llm-failed"])
    target_pool = str(payload.target_pool or "observe").strip().lower()
    if target_pool not in {"knowledge", "task", "observe"}:
        target_pool = "observe"
    return ObserveConvertDecision(
        target_pool=target_pool,
        reason_codes=[str(code).strip() for code in (payload.reason_codes or []) if str(code).strip()] or ["llm-convert"],
        confidence=float(payload.confidence or 0.0),
    )


def decide_observe_merge_or_convert(card: LiftedCard, observe_hits: list[dict[str, str]]) -> ObserveMergeDecision:
    thinking_type = get_config_str("insight.llm.route.thinking_type").strip()
    config = llm_client.LLMConfig.from_env(
        max_tokens=get_config_int("insight.llm.route.max_tokens"),
        temperature=get_config_float("insight.llm.route.temperature"),
        top_p=get_config_float("insight.llm.route.top_p"),
        extra_body={"thinking": {"type": thinking_type}} if thinking_type else None,
    )
    if config.missing_fields():
        return ObserveMergeDecision(action="keep", target_pool="observe", observe_ids=[], reason_codes=["llm-unavailable"])
    try:
        system_prompt = get_prompt("observe_merge_convert_v1.system_prompt")
        user_prompt = get_prompt("observe_merge_convert_v1.user_prompt").format(
            card_json=json.dumps(card.to_dict(), ensure_ascii=False),
            observe_hits_json=json.dumps(observe_hits[:5], ensure_ascii=False),
        )
    except Exception:
        return ObserveMergeDecision(action="keep", target_pool="observe", observe_ids=[], reason_codes=["prompt-error"])
    payload = llm_client.invoke_structured(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=llm_client.ObserveMergeConvertOutput,
    )
    if payload is None:
        return ObserveMergeDecision(action="keep", target_pool="observe", observe_ids=[], reason_codes=["llm-failed"])
    action = str(payload.action or "keep").strip().lower()
    if action not in {"keep", "merge", "convert"}:
        action = "keep"
    target_pool = str(payload.target_pool or "observe").strip().lower()
    if target_pool not in {"knowledge", "task", "observe"}:
        target_pool = "observe"
    observe_ids = [str(item).strip() for item in (payload.observe_ids or []) if str(item).strip()]
    return ObserveMergeDecision(
        action=action,
        target_pool=target_pool,
        observe_ids=observe_ids,
        reason_codes=[str(code).strip() for code in (payload.reason_codes or []) if str(code).strip()] or ["llm-observe-decision"],
        confidence=float(payload.confidence or 0.0),
    )


def optimize_card_with_llm(card: LiftedCard, hits: list[RagHit], target_pool: str) -> LiftedCard:
    thinking_type = get_config_str("insight.llm.route.thinking_type").strip()
    config = llm_client.LLMConfig.from_env(
        max_tokens=max(512, get_config_int("insight.llm.lift.max_tokens")),
        temperature=get_config_float("insight.llm.lift.temperature"),
        top_p=get_config_float("insight.llm.lift.top_p"),
        extra_body={"thinking": {"type": thinking_type}} if thinking_type else None,
    )
    if config.missing_fields():
        return card
    try:
        system_prompt = get_prompt("observe_optimize_card_v1.system_prompt")
        user_prompt = get_prompt("observe_optimize_card_v1.user_prompt").format(
            target_pool=target_pool,
            card_json=json.dumps(card.to_dict(), ensure_ascii=False),
            hits_json=json.dumps([item.to_dict() for item in hits[:5]], ensure_ascii=False),
        )
    except Exception:
        return card
    payload = llm_client.invoke_structured(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=llm_client.CardOptimizationOutput,
    )
    if payload is None:
        return card
    title = str(payload.title or "").strip() or card.title
    summary = str(payload.summary or "").strip() or card.summary
    problem = str(payload.problem or "").strip() or card.problem
    suggestion = str(payload.suggestion or "").strip() or card.suggestion
    tags = [str(item).strip() for item in (payload.tags or []) if str(item).strip()] or list(card.tags)
    confidence = float(payload.confidence) if payload.confidence is not None else float(card.confidence)
    return LiftedCard(
        card_id=card.card_id,
        candidate_id=card.candidate_id,
        title=title,
        summary=summary,
        problem=problem,
        suggestion=suggestion,
        participants=list(card.participants),
        times=str(card.times or ""),
        locations=str(card.locations or ""),
        evidence=list(card.evidence),
        tags=tags,
        confidence=confidence,
        suggested_target=target_pool if target_pool in {"knowledge", "task", "observe"} else card.suggested_target,
        source_message_ids=list(card.source_message_ids),
        topic_focus=card.topic_focus,
        message_role=card.message_role,
        context_relation=card.context_relation,
        context_evidence=list(card.context_evidence),
        decision_signals=dict(card.decision_signals),
        missing_fields=list(card.missing_fields),
    )


__all__ = [
    "ObserveAnswerResult",
    "ObserveConvertDecision",
    "ObserveMergeDecision",
    "decide_non_question_target",
    "decide_observe_merge_or_convert",
    "is_question_by_rule",
    "is_question_with_llm",
    "optimize_card_with_llm",
    "try_answer_with_rag",
]
