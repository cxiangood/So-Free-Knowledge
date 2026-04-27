from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .shared_types import InspirationCandidate, PlainMessage

_ONLY_URL_RE = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

ACTION_TERMS = (
    "需要",
    "建议",
    "请",
    "麻烦",
    "安排",
    "修复",
    "改进",
    "优化",
    "跟进",
    "截止",
    "完成",
)
IMPACT_TERMS = ("大家", "我们", "各位", "同学", "团队", "全员")
EMOTION_TERMS = ("吐槽", "抱怨", "卡住", "着急", "崩溃", "问题", "失败", "超时")


@dataclass(slots=True)
class DetectionResult:
    messages: list[PlainMessage]
    candidates: list[InspirationCandidate]


def _normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip().lower())


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


def _score_message(msg: PlainMessage, duplicate_count: int) -> tuple[dict[str, float], list[str]]:
    content = msg.content
    lowered = content.lower()

    novelty = 1.0 / float(1 + max(0, duplicate_count))
    action_hits = sum(1 for term in ACTION_TERMS if term in content)
    impact_hits = sum(1 for term in IMPACT_TERMS if term in content)
    emotion_hits = sum(1 for term in EMOTION_TERMS if term in content)

    is_question = 1.0 if ("?" in content or "？" in content) else 0.0
    has_mentions = min(1.0, len(msg.mentions) / 2.0)
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
    if "recall" in lowered:
        reasons.append("recall-mention")

    return {
        "novelty": round(novelty, 4),
        "actionability": round(actionability, 4),
        "impact": round(impact, 4),
        "emotion": round(emotion, 4),
    }, reasons


def detect_candidates(
    messages: list[PlainMessage],
    *,
    candidate_threshold: float = 0.45,
) -> DetectionResult:
    filtered: list[PlainMessage] = []
    candidates: list[InspirationCandidate] = []
    seen_count: dict[str, int] = {}

    for msg in messages:
        if _is_noise(msg.content):
            continue
        normalized = _normalize_text(msg.content)
        count = seen_count.get(normalized, 0)
        seen_count[normalized] = count + 1
        score_breakdown, reasons = _score_message(msg, count)
        score_total = (
            0.35 * score_breakdown["novelty"]
            + 0.30 * score_breakdown["actionability"]
            + 0.20 * score_breakdown["impact"]
            + 0.15 * score_breakdown["emotion"]
        )

        msg.features = {
            "normalized": normalized,
            "duplicate_count": count,
            "is_question": bool("?" in msg.content or "？" in msg.content),
            "length": len(msg.content),
        }
        filtered.append(msg)

        if score_total < candidate_threshold:
            continue

        candidate_id = "cand-" + hashlib.md5(msg.message_id.encode("utf-8")).hexdigest()[:12]
        candidates.append(
            InspirationCandidate(
                candidate_id=candidate_id,
                source_message_ids=[msg.message_id],
                score_total=round(score_total, 4),
                score_breakdown=score_breakdown,
                reasons=reasons or ["weak-signal"],
                evidence=msg.content[:220],
                content=msg.content,
            )
        )

    return DetectionResult(messages=filtered, candidates=candidates)

