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


def _clip(text: str, max_len: int) -> str:
    text = text.strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _suggest_target(candidate: InspirationCandidate, content: str) -> str:
    if candidate.score_breakdown.get("actionability", 0.0) >= 0.55:
        return "task"
    if "?" in content or "？" in content:
        return "task"
    if candidate.score_breakdown.get("novelty", 0.0) >= 0.85 and candidate.score_breakdown.get("impact", 0.0) >= 0.3:
        return "knowledge"
    return "observe"


def _mention_names(msg: MessageEvent) -> list[str]:
    names: list[str] = []
    for item in msg.mentions:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _build_template_card(candidate: InspirationCandidate, msg: MessageEvent) -> LiftedCard:
    content = candidate.content.strip()
    title = _clip(content.replace("\n", " "), 30)
    suggestion = "寤鸿鍏堝湪缇ゅ唴纭璐熻矗浜轰笌鏃堕棿鑺傜偣銆?"
    if any(term in content for term in ACTION_HINTS):
        suggestion = "寤鸿灏嗚淇″彿杞负寰呭姙骞舵寚瀹氳礋璐ｄ汉銆?"
    problem = "璁ㄨ涓嚭鐜版綔鍦ㄩ珮浠峰€煎急淇″彿銆?"
    if "?" in content or "？" in content:
        problem = "瀛樺湪鏈喅闂锛岄渶瑕佹槑纭瓟澶嶆垨琛屽姩銆?"
    audience = "鍥㈤槦鎴愬憳"
    names = _mention_names(msg)
    if names:
        audience = " ".join(f"@{name}" for name in names[:4])
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
    return LiftedCard(
        card_id=f"card-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        title=title,
        summary=_clip(content, 140),
        problem=problem,
        suggestion=suggestion,
        target_audience=audience,
        evidence=[_clip(content, 220)] if content else [],
        tags=tags,
        confidence=candidate.score_total,
        suggested_target=_suggest_target(candidate, content),  # type: ignore[arg-type]
        source_message_ids=list(candidate.source_message_ids),
    )


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


def _llm_refine_card(client: Any, card: LiftedCard) -> LiftedCard | None:
    try:
        system_prompt = get_prompt("lift.system_prompt")
    except Exception:
        system_prompt = (
            "You refine a collaboration card. Output JSON only with fields: "
            "title,summary,problem,suggestion,target_audience,tags,confidence,suggested_target. "
            "suggested_target must be one of knowledge/task/observe."
        )
    user_prompt = json.dumps(card.to_dict(), ensure_ascii=False)
    response = client.build_reply(system_prompt, user_prompt)
    if response.startswith("LLM "):
        return None
    payload = _extract_json(response)
    if not payload:
        return None
    suggested_target = str(payload.get("suggested_target", card.suggested_target)).strip().lower()
    if suggested_target not in {"knowledge", "task", "observe"}:
        suggested_target = card.suggested_target
    tags_raw = payload.get("tags", card.tags)
    tags = [str(item) for item in tags_raw if str(item).strip()] if isinstance(tags_raw, list) else card.tags
    confidence = payload.get("confidence", card.confidence)
    try:
        confidence_val = float(confidence)
    except (TypeError, ValueError):
        confidence_val = card.confidence
    return LiftedCard(
        card_id=card.card_id,
        candidate_id=card.candidate_id,
        title=_clip(str(payload.get("title", card.title)), 50),
        summary=_clip(str(payload.get("summary", card.summary)), 180),
        problem=_clip(str(payload.get("problem", card.problem)), 180),
        suggestion=_clip(str(payload.get("suggestion", card.suggestion)), 180),
        target_audience=_clip(str(payload.get("target_audience", card.target_audience)), 60),
        evidence=card.evidence,
        tags=tags or card.tags,
        confidence=max(0.0, min(1.0, confidence_val)),
        suggested_target=suggested_target,  # type: ignore[arg-type]
        source_message_ids=card.source_message_ids,
    )


def lift_candidates(
    candidates: list[InspirationCandidate],
    messages: list[MessageEvent],
    *,
    llm_max_items: int = 20,
) -> LiftResult:
    message_by_id = {item.message_id: item for item in messages}
    cards: list[LiftedCard] = []
    warnings: list[str] = []
    for candidate in candidates:
        source_id = candidate.source_message_ids[0] if candidate.source_message_ids else ""
        msg = message_by_id.get(source_id)
        if msg is None:
            msg = MessageEvent(
                event_type="im.message.receive_v1",
                event_id=f"evt-{source_id or 'unknown'}",
                create_time="",
                message_id=source_id or "unknown",
                chat_id="",
                chat_type="group",
                message_type="text",
                content_text=candidate.content,
                content_raw=candidate.content,
                root_id="",
                parent_id="",
                update_time="",
                thread_id="",
                sender_open_id="",
                sender_union_id="",
                sender_user_id="",
                sender_type="user",
                tenant_key="",
                mentions=[],
                raw={},
            )
        cards.append(_build_template_card(candidate, msg))
    if not cards:
        return LiftResult(cards=cards, warnings=warnings)

    config = LLMConfig.from_env()
    missing = config.missing_fields()
    if missing:
        warnings.append(f"LLM disabled automatically due to missing config: {', '.join(missing)}")
        return LiftResult(cards=cards, warnings=warnings)
    client = LLMClient(config)
    limit = max(0, min(len(cards), llm_max_items))
    for idx in range(limit):
        refined = _llm_refine_card(client, cards[idx])
        if refined is None:
            warnings.append(f"LLM refinement failed for {cards[idx].card_id}, fallback kept.")
            continue
        cards[idx] = refined
    return LiftResult(cards=cards, warnings=warnings)


__all__ = ["LiftResult", "lift_candidates"]
