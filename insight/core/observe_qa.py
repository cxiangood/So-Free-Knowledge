from __future__ import annotations

import logging
from dataclasses import dataclass

import llm.client as llm_client
from utils import get_config_float, get_config_int, get_config_str

from ..prompt import get_prompt
from ..shared.models import RagHit

QUESTION_MARKERS = ("?", "？", "请问", "怎么", "如何", "吗", "是否", "能否", "为什么", "求助")
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ObserveAnswerResult:
    can_answer: bool
    answer: str = ""
    hits: list[RagHit] | None = None
    reason: str = ""


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
    lines: list[str] = [f"基于知识库检索，给你一个快速答复：{query}"]
    for idx, hit in enumerate(top, start=1):
        snippet = (hit.summary or hit.text or "").replace("\n", " ").strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(f"{idx}. {hit.title or '相关知识'}：{snippet}")
    return ObserveAnswerResult(can_answer=True, answer="\n".join(lines), hits=top)


__all__ = ["ObserveAnswerResult", "is_question_by_rule", "is_question_with_llm", "try_answer_with_rag"]
