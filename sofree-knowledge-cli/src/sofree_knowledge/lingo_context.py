"""Context extraction and LLM prompt helpers for lingo candidates."""

from __future__ import annotations

import json
from typing import Any


LINGO_TYPES = {"key", "black", "confused", "nothing"}
PUBLISHABLE_LINGO_TYPES = {"key", "black"}


def extract_keyword_contexts(
    keywords: list[str],
    messages: list[dict[str, Any] | str],
    before: int = 1,
    after: int = 1,
    max_contexts: int = 30,
) -> list[dict[str, Any]]:
    """Find keyword hits and merge shared/overlapping context windows.

    ``messages`` accepts either strings or dicts with common Feishu-like fields
    such as ``message_id``, ``sender``, ``create_time`` and ``text``.
    """

    normalized_keywords = _unique_non_empty(keywords)
    normalized_messages = [_normalize_message(index, item) for index, item in enumerate(messages)]
    if not normalized_keywords or not normalized_messages:
        return []

    windows: list[dict[str, Any]] = []
    for index, message in enumerate(normalized_messages):
        text = message["text"]
        hit_keywords = [keyword for keyword in normalized_keywords if keyword in text]
        if not hit_keywords:
            continue
        windows.append(
            {
                "start": max(0, index - before),
                "end": min(len(normalized_messages) - 1, index + after),
                "keywords": hit_keywords,
            }
        )

    if not windows:
        return []

    windows.sort(key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, Any]] = []
    for window in windows:
        if merged and window["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"] = max(merged[-1]["end"], window["end"])
            merged[-1]["keywords"] = _ordered_union(
                normalized_keywords, merged[-1]["keywords"], window["keywords"]
            )
        else:
            merged.append(dict(window))

    contexts: list[dict[str, Any]] = []
    for context_id, window in enumerate(merged[:max_contexts], start=1):
        context_messages = normalized_messages[window["start"] : window["end"] + 1]
        keywords_in_context = [
            keyword
            for keyword in normalized_keywords
            if any(keyword in message["text"] for message in context_messages)
        ]
        contexts.append(
            {
                "context_id": f"ctx_{context_id}",
                "keywords": keywords_in_context,
                "message_ids": [message["message_id"] for message in context_messages],
                "text": _format_context_text(context_messages),
                "messages": context_messages,
            }
        )
    return contexts


def build_lingo_judge_prompt(
    keywords: list[str],
    contexts: list[dict[str, Any]],
) -> str:
    """Build an instruction prompt that asks the LLM for strict JSON."""

    payload = {
        "keywords": _unique_non_empty(keywords),
        "contexts": [
            {
                "context_id": context.get("context_id", ""),
                "keywords": context.get("keywords", []),
                "text": context.get("text", ""),
            }
            for context in contexts
        ],
    }
    return (
        "你是企业词典审核助手。根据给定聊天上下文判断候选关键词类型，并只输出 JSON。\n"
        "类型只能是 key、black、confused、nothing。\n"
        "key 表示值得沉淀的一般关键词或业务概念；black 表示内部黑话、缩写、项目代号或隐含含义；"
        "confused 表示证据不足或含义冲突；nothing 表示不应入库，且 value 必须为空字符串。\n"
        "如果多个关键词共用同一段上下文，可以一起判断。输出格式必须是数组，每项包含 keyword、type、value、context_ids。"
        "value 写面向词典的简短中文释义；type 为 nothing 时 value 为空字符串。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_lingo_judgements(raw: str | list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """Parse and normalize LLM JSON judgement output."""

    data: Any = raw
    if isinstance(raw, str):
        data = json.loads(_strip_code_fence(raw))

    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            data = data["items"]
        elif {"keyword", "type", "value"}.intersection(data):
            data = [data]
        else:
            data = []

    if not isinstance(data, list):
        raise ValueError("lingo judgement output must be a JSON array or object")

    judgements: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        judgement_type = str(item.get("type", "nothing")).strip().lower()
        if judgement_type not in LINGO_TYPES:
            judgement_type = "confused"
        value = str(item.get("value", "") or "").strip()
        if judgement_type == "nothing":
            value = ""
        context_ids = item.get("context_ids", [])
        if not isinstance(context_ids, list):
            context_ids = [str(context_ids)]
        judgements.append(
            {
                "keyword": str(item.get("keyword", "")).strip(),
                "type": judgement_type,
                "value": value,
                "context_ids": [str(context_id) for context_id in context_ids if context_id],
                "aliases": [
                    str(alias).strip()
                    for alias in item.get("aliases", [])
                    if str(alias).strip()
                ]
                if isinstance(item.get("aliases", []), list)
                else [],
            }
        )
    return [item for item in judgements if item["keyword"]]


def publishable_lingo_judgements(
    judgements: list[dict[str, Any]], publish_types: set[str] | None = None
) -> list[dict[str, Any]]:
    allowed_types = publish_types or PUBLISHABLE_LINGO_TYPES
    return [
        judgement
        for judgement in judgements
        if judgement.get("type") in allowed_types and str(judgement.get("value", "")).strip()
    ]


def _normalize_message(index: int, item: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "message_id": f"msg_{index + 1}",
            "sender": "",
            "create_time": "",
            "text": item,
        }

    text = item.get("text")
    if text is None:
        text = item.get("content", "")
    return {
        "message_id": str(item.get("message_id") or item.get("id") or f"msg_{index + 1}"),
        "sender": str(item.get("sender") or item.get("sender_name") or item.get("user") or ""),
        "create_time": str(item.get("create_time") or item.get("timestamp") or ""),
        "text": str(text or ""),
    }


def _format_context_text(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        prefix_parts = [
            part for part in (message.get("create_time"), message.get("sender")) if part
        ]
        prefix = " ".join(prefix_parts)
        if prefix:
            lines.append(f"[{message['message_id']}] {prefix}: {message['text']}")
        else:
            lines.append(f"[{message['message_id']}] {message['text']}")
    return "\n".join(lines)


def _unique_non_empty(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _ordered_union(order: list[str], left: list[str], right: list[str]) -> list[str]:
    values = set(left) | set(right)
    return [keyword for keyword in order if keyword in values]


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
