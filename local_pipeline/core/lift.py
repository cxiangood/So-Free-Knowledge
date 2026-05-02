from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import llm.client as llm_client

from ..prompt import get_prompt
from ..shared.models import LiftedCard

ACTION_HINTS = ("需要", "建议", "请", "安排", "修复", "优化", "跟进", "完成", "截止")
TASK_OWNER_HINTS = ("负责", "@", "同学", "团队")
TASK_TIME_HINTS = ("今天", "明天", "本周", "下周", "上午", "下午", "点", "截止")
GROUP_IMPACT_HINTS = ("大家", "我们", "各位", "团队", "全员")


@dataclass(slots=True)
class LiftResult:
    cards: list[LiftedCard]
    warnings: list[str]



def _heuristic_signals(content: str) -> dict[str, float]:
    text = content.strip()
    has_question = ("?" in text or "？" in text)
    has_action_hint = any(term in text for term in ACTION_HINTS)
    has_group_impact = any(term in text for term in GROUP_IMPACT_HINTS)
    exclaim_count = text.count("!") + text.count("！")
    emotion_score = min(1.0, exclaim_count / 3.0)
    actionability_score = 0.75 if has_action_hint else (0.55 if has_question else 0.25)
    impact_score = 0.7 if has_group_impact else 0.35
    novelty_score = 0.7
    return {
        "novelty_score": float(novelty_score),
        "actionability_score": float(actionability_score),
        "impact_score": float(impact_score),
        "emotion_score": float(emotion_score),
        "has_question": 1.0 if has_question else 0.0,
        "has_action_hint": 1.0 if has_action_hint else 0.0,
    }


def _suggest_target(content: str, decision_signals: dict[str, float], missing_fields: list[str]) -> str:
    if decision_signals.get("actionability_score", 0.0) >= 0.6:
        return "task"
    if decision_signals.get("has_question", 0.0) >= 1.0:
        return "task"
    if missing_fields:
        return "observe"
    if decision_signals.get("novelty_score", 0.0) >= 0.65 and decision_signals.get("impact_score", 0.0) >= 0.45:
        return "knowledge"
    if not content.strip():
        return "observe"
    return "observe"


def _build_default_parts(current_line: str) -> dict[str, Any]:
    content = current_line.strip()
    title = content.replace("\n", " ")[:60] if content else "消息升维卡片"
    has_question = ("?" in content or "？" in content)
    suggestion = "建议先补充关键上下文再判断后续动作。"
    if any(term in content for term in ACTION_HINTS):
        suggestion = "建议转为任务并明确负责人与时间。"
    elif has_question:
        suggestion = "建议先回答当前问题，并补充可执行信息。"
    problem = "无明显问题"
    if has_question:
        problem = "存在待解答问题"
    decision_signals = _heuristic_signals(content)
    missing_fields: list[str] = []
    if decision_signals["actionability_score"] >= 0.6:
        if not any(token in content for token in TASK_TIME_HINTS):
            missing_fields.append("time")
        if not any(token in content for token in TASK_OWNER_HINTS):
            missing_fields.append("owner")
    return {
        "title": title,
        "summary": content or "空消息",
        "suggestion": suggestion,
        "problem": problem,
        "names": [],
        "topic_focus": title[:20],
        "message_role": "question" if has_question else "new",
        "context_relation": "当前消息触发新的升维单元",
        "context_evidence": [content] if content else [],
        "decision_signals": decision_signals,
        "missing_fields": missing_fields,
    }


def _try_llm_parts(*, current_line: str, context_lines: list[str]) -> dict[str, Any] | None:
    config = llm_client.LLMConfig.from_env(max_tokens=2048, temperature=0.0, top_p=0.1,extra_body={"thinking": {"type": "disabled"}},)
    if config.missing_fields():
        return None
    try:
        system_prompt = get_prompt("lift_v2.system_prompt")
        user_prompt = get_prompt("lift_v2.user_prompt").format(
            current_line=current_line,
            context_lines="\n".join(context_lines) if context_lines else "",
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
    return {
        "title": payload.title,
        "summary": payload.summary,
        "suggestion": payload.suggestion,
        "problem": payload.problem,
        "names": payload.names,
        "topic_focus": payload.topic_focus,
        "message_role": payload.message_role,
        "context_relation": payload.context_relation,
        "context_evidence": payload.context_evidence,
        "decision_signals": payload.decision_signals,
        "missing_fields": payload.missing_fields,
    }


def lift_candidates(messages: list[str], *, llm_max_items: int = 20) -> LiftResult:
    del llm_max_items
    cards: list[LiftedCard] = []
    warnings: list[str] = []
    if not messages:
        return LiftResult(cards=cards, warnings=warnings)

    current_line = str(messages[-1]).strip()
    context_lines = [str(item) for item in messages[:-1]]
    defaults = _build_default_parts(current_line)
    llm_parts = _try_llm_parts(current_line=current_line, context_lines=context_lines)
    parts = dict(defaults)
    if llm_parts is None:
        warnings.append("LLM card compose failed, fallback kept.")
    else:
        parts.update(llm_parts)

    content = current_line
    decision_signals_raw = parts.get("decision_signals", {})
    if not isinstance(decision_signals_raw, dict):
        decision_signals_raw = {}
    decision_signals = {str(k): float(v) for k, v in decision_signals_raw.items() if isinstance(v, (int, float))}
    missing_fields = [str(x) for x in (parts.get("missing_fields", []) or []) if str(x).strip()]
    suggested_target = _suggest_target(content, decision_signals, missing_fields)

    tags: list[str] = []
    if decision_signals.get("novelty_score", 0.0) >= 0.7:
        tags.append("novel-content")
    if decision_signals.get("actionability_score", 0.0) >= 0.6:
        tags.append("actionable-signal")
    if decision_signals.get("impact_score", 0.0) >= 0.5:
        tags.append("group-impact")
    if decision_signals.get("emotion_score", 0.0) >= 0.5:
        tags.append("emotion-intensity")
    if not tags:
        tags = ["weak-signal"]

    confidence = max(
        0.3,
        min(
            0.95,
            0.2
            + 0.35 * decision_signals.get("novelty_score", 0.0)
            + 0.35 * decision_signals.get("actionability_score", 0.0)
            + 0.2 * decision_signals.get("impact_score", 0.0)
            + 0.1 * decision_signals.get("emotion_score", 0.0),
        ),
    )

    card = LiftedCard(
        card_id="card-current",
        candidate_id="cand-current",
        title=str(parts.get("title", defaults["title"])),
        summary=str(parts.get("summary", defaults["summary"])),
        problem=str(parts.get("problem", defaults["problem"])),
        suggestion=str(parts.get("suggestion", defaults["suggestion"])),
        target_audience=str(parts.get("names", defaults["names"])),
        evidence=[content] if content else [],
        tags=tags,
        confidence=float(confidence),
        suggested_target=suggested_target,
        source_message_ids=["current"],
        topic_focus=str(parts.get("topic_focus", defaults["topic_focus"])),
        message_role=str(parts.get("message_role", defaults["message_role"])),
        context_relation=str(parts.get("context_relation", defaults["context_relation"])),
        context_evidence=[str(x) for x in (parts.get("context_evidence", defaults["context_evidence"]) or []) if str(x).strip()],
        decision_signals=decision_signals,
        missing_fields=missing_fields,
    )
    cards.append(card)
    return LiftResult(cards=cards, warnings=warnings)


__all__ = ["LiftResult", "lift_candidates"]

if __name__ == "__main__":
    test_messages = [
        "[Alice] 大家好",
        "[Alice] 今天下午需要开会",
        "[Alice] 1点种",
        "[Alice] 在我办公室",  # duplicate
        "[Alice] 张三和李四需要来开会",  # duplicate
        "[Alice] 王五也要来"
        "[Bob] 午饭的牛肉真好吃",
        "[Alice] 赵六也来听一下",
        "[李四] 我要来吗？",
        #"[Peter] 笑死我了哈哈哈哈"
        "[王五] 在哪开会？",
    ]
    result = lift_candidates(test_messages)
    print("Lift End")
    print(result)
