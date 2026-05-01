from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import llm.client as llm_client

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
    has_question = ("?" in content or "？" in content)
    message_role = "question" if has_question else "new"
    decision_signals = {
        "novelty_score": float(candidate.score_breakdown.get("novelty", 0.0)),
        "actionability_score": float(candidate.score_breakdown.get("actionability", 0.0)),
        "impact_score": float(candidate.score_breakdown.get("impact", 0.0)),
        "emotion_score": float(candidate.score_breakdown.get("emotion", 0.0)),
        "has_question": 1.0 if has_question else 0.0,
        "has_action_hint": 1.0 if any(term in content for term in ACTION_HINTS) else 0.0,
    }
    missing_fields: list[str] = []
    if decision_signals["actionability_score"] >= 0.55:
        if not any(token in content for token in ("今天", "明天", "本周", "下午", "上午", "点", "截止")):
            missing_fields.append("time")
        if not any(token in content for token in ("负责", "@", "同学", "团队")):
            missing_fields.append("owner")
    return {
        "title": title,
        "summary": summary,
        "suggestion": suggestion,
        "problem": problem,
        "names": names,
        "topic_focus": title,
        "message_role": message_role,
        "context_relation": "当前消息触发新信号",
        "context_evidence": [content] if content else [],
        "decision_signals": decision_signals,
        "missing_fields": missing_fields,
    }


def _try_llm_parts(
    candidate: InspirationCandidate,
    current_line: str | None,
    context_lines: list[str],
) -> dict[str, Any] | None:
    # 语义提升任务：输出固定JSON格式，包含几个短文本字段，使用较快参数
    config = llm_client.LLMConfig.from_env(max_tokens=768, temperature=0.1, top_p=0.2)
    if config.missing_fields():
        return None
    try:
        system_prompt = get_prompt("lift.system_prompt")
        context_text = _format_context_lines(context_lines)
        user_prompt = get_prompt("lift.user_prompt").format(
            candidate_score_breakdown=json.dumps(candidate.score_breakdown, ensure_ascii=False),
            current_line=(current_line or ""),
            context_lines=context_text,
        )
    except Exception:
        return None
    payload = llm_client.invoke_structured(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=llm_client.LiftParts,
    )
    if payload is None:
        return None
    title = payload.title
    summary = payload.summary
    suggestion = payload.suggestion
    problem = payload.problem
    names = payload.names
    topic_focus = str(getattr(payload, "topic_focus", "") or "")
    message_role = str(getattr(payload, "message_role", "") or "")
    context_relation = str(getattr(payload, "context_relation", "") or "")
    context_evidence = [str(x).strip() for x in (getattr(payload, "context_evidence", []) or []) if str(x).strip()]
    raw_signals = getattr(payload, "decision_signals", {}) or {}
    if not isinstance(raw_signals, dict):
        raw_signals = {}
    decision_signals = {str(k): float(v) for k, v in raw_signals.items() if isinstance(v, (int, float))}
    missing_fields = [str(x).strip() for x in (getattr(payload, "missing_fields", []) or []) if str(x).strip()]
    if not all([title, summary, suggestion, problem]):
        return None
    return {
        "title": title,
        "summary": summary,
        "suggestion": suggestion,
        "problem": problem,
        "names": names,
        "topic_focus": topic_focus,
        "message_role": message_role,
        "context_relation": context_relation,
        "context_evidence": context_evidence,
        "decision_signals": decision_signals,
        "missing_fields": missing_fields,
    }


def _format_context_lines(context_lines: list[str]) -> str:
    if not context_lines:
        return "(empty)"
    rows: list[str] = []
    for idx, line in enumerate(context_lines, start=1):
        text = str(line).replace("\n", " ").strip()
        rows.append(f"{idx}. {text}")
    return "\n".join(rows)


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
        topic_focus=str(parts.get("topic_focus", defaults["topic_focus"])),
        message_role=str(parts.get("message_role", defaults["message_role"])),
        context_relation=str(parts.get("context_relation", defaults["context_relation"])),
        context_evidence=[str(x) for x in parts.get("context_evidence", defaults["context_evidence"]) if str(x).strip()],
        decision_signals={
            str(k): float(v)
            for k, v in (
                parts.get("decision_signals", defaults["decision_signals"])
                if isinstance(parts.get("decision_signals", defaults["decision_signals"]), dict)
                else defaults["decision_signals"]
            ).items()
            if isinstance(v, (int, float))
        },
        missing_fields=[str(x) for x in parts.get("missing_fields", defaults["missing_fields"]) if str(x).strip()],
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
