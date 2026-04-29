from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm.client import LLMClient, LLMConfig

from ..msg.types import MessageEvent
from ..prompt import get_prompt
from ..shared.models import InspirationCandidate, LiftedCard

ACTION_HINTS = ("需要", "建议", "请", "安排", "修复", "优化", "跟进", "完成", "截止")


@dataclass(slots=True)
class LiftResult:
    cards: list[LiftedCard]
    warnings: list[str]


def _suggest_target(candidate: InspirationCandidate, content: str) -> str:
    if candidate.score_breakdown.get("actionability", 0.0) >= 0.55:
        return "task"
    if "?" in content or "？" in content:
        return "task"
    if candidate.score_breakdown.get("novelty", 0.0) >= 0.85 and candidate.score_breakdown.get("impact", 0.0) >= 0.3:
        return "knowledge"
    return "observe"


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _build_default_parts(candidate: InspirationCandidate, current_message: MessageEvent | None) -> dict[str, Any]:
    content = candidate.content.strip()
    title = content.replace("\n", " ")
    suggestion = "建议先在群内确认负责人与时间节点。"
    if any(term in content for term in ACTION_HINTS):
        suggestion = "建议将该信号转为待办并指定负责人。"
    problem = "讨论中出现潜在高价值弱信号。"
    if "?" in content or "？" in content:
        problem = "存在未决问题，需要明确答复或行动。"
    names = []
    summary = content
    return {
        "title": title,
        "summary": summary,
        "suggestion": suggestion,
        "problem": problem,
        "names": names,
    }


def _try_llm_parts(
    candidate: InspirationCandidate,
    current_line: str | None,
    context_lines: list[str],
) -> dict[str, Any] | None:
    config = LLMConfig.from_env(max_tokens=300, temperature=0.2)
    if config.missing_fields():
        return None
    try:
        system_prompt = get_prompt("lift.system_prompt")
        user_prompt = get_prompt("lift.user_prompt").format(
            candidate_score_breakdown=json.dumps(candidate.score_breakdown, ensure_ascii=False),
            current_line=current_line,
            context_lines=context_lines
        )
    except Exception:
        return None
    response = LLMClient(config).build_reply(system_prompt, user_prompt)
    if response.startswith("LLM "):
        return None
    payload = _extract_json(response)
    if not payload:
        return None
    title = payload.get("title", "")
    summary = payload.get("summary", "")
    suggestion = payload.get("suggestion", "")
    problem = payload.get("problem", "")
    names = payload.get("names", [])
    if not all([title, summary, suggestion, problem]):
        return None
    if not isinstance(names, list):
        return None
    return {
        "title": title,
        "summary": summary,
        "suggestion": suggestion,
        "problem": problem,
        "names": names,
    }


def _build_card_with_llm_fallback(
    candidate: InspirationCandidate,
    current_line: str | None,
    context_lines: list[str],
) -> tuple[LiftedCard, str | None]:
    defaults = _build_default_parts(candidate, current_line)
    llm_parts = _try_llm_parts(candidate, current_line, context_lines)
    warning: str | None = None
    if llm_parts is None:
        parts = defaults
        warning = f"LLM card compose failed for card-{candidate.candidate_id}, fallback kept."
    else:
        parts = dict(defaults)
        parts.update(llm_parts)

    content = candidate.content.strip()
    names = parts.get("names", [])
    if isinstance(names, list) and names:
        audience = " ".join(f"@{name}" for name in names if str(name).strip())
    else:
        audience = "团队成员"

    tags: list[str] = []
    if candidate.score_breakdown.get("novelty", 0.0) >= 0.9:
        tags.append("novel-content")
    if candidate.score_breakdown.get("actionability", 0.0) >= 0.55:
        tags.append("actionable-signal")
    if candidate.score_breakdown.get("impact", 0.0) >= 0.45:
        tags.append("group-impact")
    if candidate.score_breakdown.get("emotion", 0.0) >= 0.45:
        tags.append("emotion-intensity")
    if not tags:
        tags = ["weak-signal"]

    card = LiftedCard(
        card_id=f"card-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        title=str(parts.get("title", defaults["title"])), 
        summary=str(parts.get("summary", defaults["summary"])), 
        problem=str(parts.get("problem", defaults["problem"])), 
        suggestion=str(parts.get("suggestion", defaults["suggestion"])),
        target_audience=audience,
        evidence=[content] if content else [],
        tags=tags,
        confidence=candidate.score_total,
        suggested_target=_suggest_target(candidate, content),  # type: ignore[arg-type]
        source_message_ids=list(candidate.source_message_ids),
    )
    return card, warning


def lift_candidates(
    candidates: list[InspirationCandidate],
    messages: list[str],
    *,
    llm_max_items: int = 20,
) -> LiftResult:
    del llm_max_items  # compatibility only; first-candidate strategy always builds at most one card.

    cards: list[LiftedCard] = []
    warnings: list[str] = []
    if not candidates:
        return LiftResult(cards=cards, warnings=warnings)

    candidate = candidates[0]
    current_message = messages[-1] if messages else None
    context_messages = messages[:-1] if messages else []
    card, warning = _build_card_with_llm_fallback(candidate, current_message, context_messages)
    cards.append(card)
    if warning:
        warnings.append(warning)
    return LiftResult(cards=cards, warnings=warnings)


__all__ = ["LiftResult", "lift_candidates"]
