from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from llm.client import LLMClient, LLMConfig

from ..prompt import get_prompt
from ..shared.models import RagHit

QUESTION_MARKERS = ("?", "？", "请问", "怎么", "如何", "吗", "是否", "能否", "为什么", "求助")


@dataclass(slots=True)
class ObserveAnswerResult:
    can_answer: bool
    answer: str = ""
    hits: list[RagHit] | None = None
    reason: str = ""


def is_question_by_rule(*, summary: str, problem: str, content: str) -> bool:
    text = f"{summary}\n{problem}\n{content}".lower()
    return any(marker in text for marker in QUESTION_MARKERS)


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


def is_question_with_llm(*, summary: str, problem: str, content: str) -> bool:
    rule_guess = is_question_by_rule(summary=summary, problem=problem, content=content)
    config = LLMConfig.from_env(max_tokens=120, temperature=0.0)
    if config.missing_fields():
        return rule_guess
    try:
        system_prompt = get_prompt("observe_qa.system_prompt")
        user_prompt = get_prompt("observe_qa.user_prompt").format(summary=summary, problem=problem, content=content)
    except Exception:
        return rule_guess
    response = LLMClient(config).build_reply(system_prompt, user_prompt)
    if response.startswith("LLM "):
        return rule_guess
    payload = _extract_json(response)
    if not payload:
        return rule_guess
    value = payload.get("is_question")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
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
