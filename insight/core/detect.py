from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import llm.client as llm_client
from utils import get_config_float, get_config_int, get_config_str

from ..prompt import get_prompt

_ONLY_URL_RE = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)
_SIMPLE_MSG_RE = re.compile(r"^\s*\[(?P<sender>[^\]]*)\]\s*(?P<content>.*)$")

ACTION_TERMS = ("需要", "建议", "请", "安排", "修复", "改进", "优化", "跟进", "截止", "完成")
IMPACT_TERMS = ("大家", "我们", "各位", "同学", "团队", "全员")
EMOTION_TERMS = ("吐槽", "抱怨", "卡住", "着急", "崩溃", "问题", "失败", "超时")

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectionResult:
    messages: list[str]
    value_score: float


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


def _message_content(text: str) -> str:
    match = _SIMPLE_MSG_RE.match(text)
    if match:
        return match.group("content").strip()
    return text.strip()


def _rule_value_score(content: str) -> float:
    novelty = 1.0
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
    score_breakdown = {
        "novelty": float(novelty),
        "actionability": float(actionability),
        "impact": float(impact),
        "emotion": float(emotion),
    }

    score_total = (
        0.25 * score_breakdown["novelty"]
        + 0.25 * score_breakdown["actionability"]
        + 0.25 * score_breakdown["impact"]
        + 0.25 * score_breakdown["emotion"]
    )
    return float(max(0.0, min(100.0, score_total * 100.0)))


def _detect_with_llm(*, context_lines: list[str], current_line: str) -> float | None:
    thinking_type = get_config_str("insight.llm.detect.thinking_type").strip()
    config = llm_client.LLMConfig.from_env(
        max_tokens=get_config_int("insight.llm.detect.max_tokens"),
        temperature=get_config_float("insight.llm.detect.temperature"),
        top_p=get_config_float("insight.llm.detect.top_p"),
        extra_body={"thinking": {"type": thinking_type}} if thinking_type else None,
    )
    if config.missing_fields():
        return None
    try:
        system_prompt = get_prompt("detect_v2.system_prompt")
        user_prompt = get_prompt("detect_v2.user_prompt").format(current_line=current_line, context_lines="\n".join(context_lines))
    except Exception:
        return None
    payload = llm_client.invoke_structured(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=llm_client.DetectValueScore,
    )
    if payload is None:
        return None
    return float(max(0.0, min(100.0, payload.value_score)))


def detect_candidates(messages: list[str], *, message_id: str = "", chat_id: str = "") -> DetectionResult:
    filtered = [msg for msg in messages if not _is_noise(msg)]
    if not filtered:
        return DetectionResult(messages=[], value_score=0.0)

    current_content = _message_content(filtered[-1])
    context_raw = filtered[:-1]
    value_score = _detect_with_llm(context_lines=context_raw, current_line=current_content)
    if value_score is None:
        LOGGER.warning(
            "fallback module=signal_detect reason=llm_failed strategy=rule_scoring message_id=%s chat_id=%s",
            message_id or "-",
            chat_id or "-",
        )
        value_score = _rule_value_score(current_content)

    return DetectionResult(messages=filtered, value_score=float(max(0.0, min(100.0, value_score))))


__all__ = ["DetectionResult", "detect_candidates"]
