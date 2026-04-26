"""Rule-based confused detection + LLM judgement helpers."""

from __future__ import annotations

import json
import re
from typing import Any


CONFUSED_PHRASE_PATTERNS = (
    r"什么[?？]*$",
    r"啥意思[?？]*$",
    r"没懂",
    r"啥",
    r"不懂",
    r"没明白",
    r"不明白",
    r"没看懂",
    r"看不懂",
    r"听不懂",
    r"不太明白",
    r"求解释",
    r"能详细说",
    r"^[?？]+$",
)

DEFAULT_CONFUSED_REACTION_KEYS = {
    "question",
    "question_mark",
    "what",
    "dizzy",
    "doubt",
    "confused",
    "疑问",
    "什么",
    "?",
    "？",
}


def detect_confused_candidates(
    messages: list[dict[str, Any] | str],
    target_message_id: str = "",
    reactions: list[dict[str, Any]] | None = None,
    confused_reaction_keys: set[str] | None = None,
    max_followup_gap: int = 3,
    max_candidates: int = 20,
) -> list[dict[str, Any]]:
    """Detect candidate confused events from chat logs."""

    normalized_messages = [_normalize_message(index, item) for index, item in enumerate(messages)]
    if not normalized_messages:
        return []

    index_by_id = {item["message_id"]: index for index, item in enumerate(normalized_messages) if item["message_id"]}
    target = str(target_message_id).strip()

    buckets: dict[str, dict[str, Any]] = {}
    for index, message in enumerate(normalized_messages):
        text = message["text"].strip()
        if not text:
            continue
        phrase_hit = _contains_confused_phrase(text)
        anchor_id = _infer_anchor_message_id(
            message=message,
            message_index=index,
            messages=normalized_messages,
            index_by_id=index_by_id,
            max_followup_gap=max_followup_gap,
        )
        if not anchor_id:
            continue
        if target and anchor_id != target:
            continue

        if phrase_hit:
            _upsert_candidate(
                buckets=buckets,
                anchor_id=anchor_id,
                trigger="confused_phrase",
                score=0.7,
                evidence={
                    "type": "message",
                    "message_id": message["message_id"],
                    "text": text,
                    "reply_to": message["parent_id"] or message["root_id"],
                },
                context=_build_context_window(anchor_id, normalized_messages, index_by_id),
            )

        if anchor_id in {message.get("parent_id", ""), message.get("root_id", "")}:
            _upsert_candidate(
                buckets=buckets,
                anchor_id=anchor_id,
                trigger="reply_or_thread_followup",
                score=0.25,
                evidence={
                    "type": "reply",
                    "message_id": message["message_id"],
                    "text": text,
                },
                context=_build_context_window(anchor_id, normalized_messages, index_by_id),
            )

    reaction_keys = {
        key.strip().lower()
        for key in (confused_reaction_keys or DEFAULT_CONFUSED_REACTION_KEYS)
        if key and key.strip()
    }

    for reaction in reactions or []:
        anchor_id = str(reaction.get("message_id", "")).strip()
        if not anchor_id or (target and anchor_id != target):
            continue
        if anchor_id not in index_by_id:
            continue
        token = _normalize_reaction_key(reaction)
        if token and _is_confused_reaction_key(token, reaction_keys):
            _upsert_candidate(
                buckets=buckets,
                anchor_id=anchor_id,
                trigger="confused_reaction",
                score=0.8,
                evidence={
                    "type": "reaction",
                    "message_id": anchor_id,
                    "reaction": token,
                    "user_id": str(reaction.get("user_id", "")),
                },
                context=_build_context_window(anchor_id, normalized_messages, index_by_id),
            )

    candidates = sorted(
        buckets.values(),
        key=lambda item: (item["score"], item["target_message_id"]),
        reverse=True,
    )
    return candidates[:max_candidates]


def build_confused_judge_prompt(candidate: dict[str, Any]) -> str:
    payload = {
        "candidate": {
            "target_message_id": str(candidate.get("target_message_id", "")),
            "score": float(candidate.get("score", 0.0)),
            "triggers": candidate.get("triggers", []),
            "evidence": candidate.get("evidence", []),
            "context_messages": candidate.get("context_messages", []),
        }
    }
    return (
        "你是群聊理解辅助器。请判断 candidate 是否代表“用户对上文存在理解障碍”。\n"
        "输出必须是 JSON 对象，且只输出 JSON，不要额外文字。\n"
        "字段要求：\n"
        '- is_confused: 布尔值\n'
        '- confidence: 0 到 1 的小数\n'
        '- reason: 简短中文理由（不超过 40 字）\n'
        '- micro_explain: 若 is_confused=true，给出“无感小插入解释”，不超过 60 字；'
        "像补充说明，不要像 bot 回复，不要称呼用户，不要用“你可以/请”。若 false 置空字符串。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_confused_judgement(raw: str | dict[str, Any]) -> dict[str, Any]:
    data: Any = raw
    if isinstance(raw, str):
        data = json.loads(_strip_code_fence(raw))
    if not isinstance(data, dict):
        raise ValueError("confused judgement output must be a JSON object")

    is_confused = bool(data.get("is_confused", False))
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason", "") or "").strip()
    micro_explain = str(data.get("micro_explain", "") or "").strip() if is_confused else ""
    return {
        "is_confused": is_confused,
        "confidence": confidence,
        "reason": reason,
        "micro_explain": micro_explain,
    }


def format_inline_explanation(micro_explain: str) -> str:
    text = str(micro_explain or "").strip()
    if not text:
        return ""
    wrapped = text
    if not wrapped.startswith("（"):
        wrapped = f"（{wrapped}"
    if not wrapped.endswith("）"):
        wrapped = f"{wrapped}）"
    return wrapped


def _normalize_message(index: int, item: dict[str, Any] | str) -> dict[str, str]:
    if isinstance(item, str):
        return {
            "message_id": f"msg_{index + 1}",
            "text": item,
            "parent_id": "",
            "root_id": "",
            "sender_id": "",
            "create_time": "",
        }

    text = item.get("text")
    if text is None:
        text = item.get("content", "")
    sender = item.get("sender", {})
    sender_id = ""
    if isinstance(sender, dict):
        sender_id = str(sender.get("id") or sender.get("sender_id") or sender.get("user_id") or "")
    elif sender:
        sender_id = str(sender)
    return {
        "message_id": str(item.get("message_id") or item.get("id") or f"msg_{index + 1}"),
        "text": str(text or ""),
        "parent_id": str(item.get("parent_id") or item.get("reply_to") or ""),
        "root_id": str(item.get("root_id") or ""),
        "sender_id": sender_id,
        "create_time": str(item.get("create_time") or item.get("timestamp") or ""),
    }


def _contains_confused_phrase(text: str) -> bool:
    normalized = text.strip().lower()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in CONFUSED_PHRASE_PATTERNS)


def _infer_anchor_message_id(
    message: dict[str, str],
    message_index: int,
    messages: list[dict[str, str]],
    index_by_id: dict[str, int],
    max_followup_gap: int,
) -> str:
    parent_id = message.get("parent_id", "")
    root_id = message.get("root_id", "")
    if parent_id and parent_id in index_by_id:
        return parent_id
    if root_id and root_id in index_by_id:
        return root_id
    if message_index <= 0:
        return ""

    text = message.get("text", "").strip()
    if _contains_confused_phrase(text):
        for back in range(1, max_followup_gap + 1):
            prev_index = message_index - back
            if prev_index < 0:
                break
            previous = messages[prev_index]
            if previous.get("sender_id") and previous.get("sender_id") == message.get("sender_id"):
                continue
            return previous.get("message_id", "")
    return ""


def _build_context_window(
    anchor_id: str,
    messages: list[dict[str, str]],
    index_by_id: dict[str, int],
) -> list[dict[str, str]]:
    if anchor_id not in index_by_id:
        return []
    anchor_index = index_by_id[anchor_id]
    start = max(0, anchor_index - 1)
    end = min(len(messages) - 1, anchor_index + 2)
    return [
        {
            "message_id": messages[index]["message_id"],
            "text": messages[index]["text"],
            "sender_id": messages[index]["sender_id"],
            "create_time": messages[index]["create_time"],
        }
        for index in range(start, end + 1)
    ]


def _upsert_candidate(
    buckets: dict[str, dict[str, Any]],
    anchor_id: str,
    trigger: str,
    score: float,
    evidence: dict[str, Any],
    context: list[dict[str, str]],
) -> None:
    if anchor_id not in buckets:
        buckets[anchor_id] = {
            "target_message_id": anchor_id,
            "score": 0.0,
            "triggers": [],
            "evidence": [],
            "context_messages": context,
        }
    candidate = buckets[anchor_id]
    candidate["score"] = min(1.0, candidate["score"] + score)
    if trigger not in candidate["triggers"]:
        candidate["triggers"].append(trigger)
    candidate["evidence"].append(evidence)
    if not candidate["context_messages"] and context:
        candidate["context_messages"] = context


def _normalize_reaction_key(reaction: dict[str, Any]) -> str:
    for key in ("reaction_key", "reaction_type", "reaction", "type", "name", "text"):
        value = reaction.get(key)
        if value:
            return str(value).strip().lower()
    reaction_type = reaction.get("reaction_type")
    if isinstance(reaction_type, dict):
        for key in ("key", "name", "emoji_type", "text"):
            value = reaction_type.get(key)
            if value:
                return str(value).strip().lower()
    return ""


def _is_confused_reaction_key(token: str, allowed_keys: set[str]) -> bool:
    lowered = token.lower()
    if lowered in allowed_keys:
        return True
    return lowered in {"?", "？"}


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
