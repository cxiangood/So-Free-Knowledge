from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from llm.client import LLMClient, LLMConfig

from ..shared.models import InspirationCandidate
from ..prompt import get_prompt

_ONLY_URL_RE = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_SIMPLE_MSG_RE = re.compile(r"^\s*\[(?P<sender>[^\]]*)\]\s*(?P<content>.*)$")

ACTION_TERMS = ("需要", "建议", "请", "安排", "修复", "改进", "优化", "跟进", "截止", "完成")
IMPACT_TERMS = ("大家", "我们", "各位", "同学", "团队", "全员")
EMOTION_TERMS = ("吐槽", "抱怨", "卡住", "着急", "崩溃", "问题", "失败", "超时")


@dataclass(slots=True)
class DetectionResult:
    messages: list[str]
    candidates: list[InspirationCandidate]



def _is_noise(text: str) -> bool:
    content = text.strip()
    if not content:
        return True
    lowered = content.lower()
    if lowered == "this message was recalled":
        return True
    if len(content) <= 1:
        return True
    if _ONLY_URL_RE.match(content):
        return True
    return False


def _score_message_rule(content: str, duplicate_count: int) -> tuple[dict[str, float], list[str]]:
    novelty = 1.0 / float(1 + max(0, duplicate_count))
    action_hits = sum(1 for term in ACTION_TERMS if term in content)
    impact_hits = sum(1 for term in IMPACT_TERMS if term in content)
    emotion_hits = sum(1 for term in EMOTION_TERMS if term in content)
    is_question = 1.0 if ("?" in content or "？" in content) else 0.0
    mention_hits = max(0, content.count("@"))
    has_mentions = min(1.0, mention_hits / 2.0)
    exclaim_strength = min(1.0, (content.count("!") + content.count("！")) / 3.0)
    actionability = min(1.0, 0.35 * is_question + 0.45 * min(1.0, action_hits / 2.0) + 0.2 * has_mentions)
    impact = min(1.0, 0.5 * has_mentions + 0.5 * min(1.0, impact_hits / 2.0))
    emotion = min(1.0, 0.7 * min(1.0, emotion_hits / 2.0) + 0.3 * exclaim_strength)

    reasons: list[str] = []
    if novelty >= 0.9:
        reasons.append("novel-content")
    if actionability >= 0.55:
        reasons.append("actionable-signal")
    if impact >= 0.45:
        reasons.append("group-impact")
    if emotion >= 0.45:
        reasons.append("emotion-intensity")
    return {
        "novelty": round(novelty, 4),
        "actionability": round(actionability, 4),
        "impact": round(impact, 4),
        "emotion": round(emotion, 4),
    }, reasons


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


def _score_total(score_breakdown: dict[str, float]) -> float:
    return (
        0.25 * score_breakdown["novelty"]
        + 0.25 * score_breakdown["actionability"]
        + 0.25 * score_breakdown["impact"]
        + 0.25 * score_breakdown["emotion"]
    )


def _score_with_llm(*, context_lines: list[str], current_line: str) -> dict[str, float] | None:
    # 信号检测任务：输出固定JSON格式，只需4个数值，使用最快参数
    config = LLMConfig.from_env(max_tokens=128, temperature=0.0, top_p=0.1)
    if config.missing_fields():
        return None
    client = LLMClient(config)
    try:
        system_prompt = get_prompt("detect.system_prompt")
        user_prompt = get_prompt("detect.user_prompt").format(current_line=current_line, context_lines="\n".join(context_lines))
    except Exception:
        return None
    
    response = client.build_reply(system_prompt, user_prompt)
    if response.startswith("LLM "):
        return None
    payload = _extract_json(response)
    if not payload:
        return None
    score_breakdown = {
        "novelty": payload.get("novelty")/100,
        "actionability": payload.get("actionability")/100,
        "impact": payload.get("impact")/100,
        "emotion": payload.get("emotion")/100,
    }
    return score_breakdown


def detect_candidates(messages: list[str], *, candidate_threshold: float = 0.45) -> DetectionResult:
    filtered = [msg for msg in messages if not _is_noise(msg)]
    if not filtered:
        return DetectionResult(messages=[], candidates=[])

    current_content = filtered[-1]
    

    context_raw = filtered[:-1]

    scored = _score_with_llm(context_lines=context_raw, current_line=current_content)
    if scored is None:
        score_breakdown, _ = _score_message_rule(current_content, 0)
    else:
        score_breakdown = scored

    score_total = _score_total(score_breakdown)
    if score_total < candidate_threshold:
        return DetectionResult(messages=filtered, candidates=[])

    candidate_id = "cand-" + hashlib.md5(current_content.encode("utf-8")).hexdigest()[:12]
    candidate = InspirationCandidate(
        candidate_id=candidate_id,
        source_message_ids=[candidate_id],
        score_total=round(score_total, 4),
        score_breakdown=score_breakdown,
        content=current_content,
    )
    return DetectionResult(messages=filtered, candidates=[candidate])
__all__ = ["DetectionResult", "detect_candidates"]
    
if __name__ == "__main__":
    test_messages = [
        "[Alice] 大家好！",
        "[Bob] 需要安排一下下周的会议。",
        "[Charlie] 这个功能太棒了！",
        "[Alice] 大家好！",  # duplicate
        # "[Bob] 需要安排一下下周的会议。",  # duplicate
        # "[Dave] @Alice 我觉得这个问题很严重，需要尽快修复！",
    ]
    result = detect_candidates(test_messages, candidate_threshold=0.01)
    print("Detect End")