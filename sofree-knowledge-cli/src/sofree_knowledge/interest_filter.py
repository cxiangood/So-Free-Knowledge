"""OpenClaw prompt helpers for garbage-message filtering in interest digest."""

from __future__ import annotations

import json
from typing import Any


def build_interest_filter_prompt(
    messages: list[dict[str, Any] | str],
    interests: list[str] | None = None,
    max_messages: int = 100,
) -> str:
    normalized = [_normalize_message(index, item) for index, item in enumerate(messages)][: max(1, int(max_messages))]
    payload = {
        "task": "filter_interest_garbage_messages",
        "interests": [str(item).strip() for item in (interests or []) if str(item).strip()],
        "messages": normalized,
    }
    return (
        "你是群聊消息筛选器。目标：从 messages 中筛掉垃圾消息（系统提醒、自动通知、灌水、闲聊、无行动价值）。\n"
        "请按 message_id 输出逐条判断，必须只输出 JSON，不要额外文字。\n"
        "输出必须是 JSON 数组，每个元素字段如下：\n"
        '- message_id: 字符串\n'
        '- include_in_digest: 布尔值；是否应出现在兴趣摘要\n'
        '- is_garbage: 布尔值；是否为垃圾/低价值消息\n'
        '- importance: 0~1 小数；对兴趣摘要的价值强度（按下面评分规则计算）\n'
        '- score_impact: 0~1；业务影响（决策/风险/客户影响/收入影响）\n'
        '- score_actionability: 0~1；可执行性（是否有明确行动、负责人、截止时间）\n'
        '- score_timeliness: 0~1；时效性（紧急程度、是否近期需要处理）\n'
        '- score_relevance: 0~1；与 interests 的相关性\n'
        '- reason: 不超过 30 字，简述判断依据\n'
        '- summary: 不超过 60 字；若 include_in_digest=true 给出可展示摘要，否则空字符串\n'
        "判断标准：\n"
        "1) 优先看是否包含明确业务信息（需求、风险、上线、截止、故障、决策、待办、负责人、时间点）。\n"
        "2) 即使命中兴趣词，但若只是处罚提醒/系统文案/模板噪声，也应判为垃圾。\n"
        "3) importance = 0.35*score_impact + 0.30*score_actionability + 0.20*score_timeliness + 0.15*score_relevance。\n"
        "4) summary 必须去口头废话，只保留事实与动作。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_interest_filter_judgements(raw: str | list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    data: Any = raw
    if isinstance(raw, str):
        data = json.loads(_strip_code_fence(raw))

    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            data = data["items"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError("interest filter judgement output must be a JSON array/object")

    result: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        message_id = str(item.get("message_id") or item.get("id") or "").strip()
        if not message_id:
            continue
        include_in_digest = _coerce_bool(item.get("include_in_digest"), default=False)
        is_garbage = _coerce_bool(item.get("is_garbage"), default=not include_in_digest)
        importance = _coerce_float(item.get("importance"), default=0.0)
        importance = max(0.0, min(1.0, importance))
        reason = str(item.get("reason") or "").strip()
        summary = str(item.get("summary") or "").strip() if include_in_digest else ""
        result.append(
            {
                "message_id": message_id,
                "include_in_digest": include_in_digest,
                "is_garbage": is_garbage,
                "importance": importance,
                "score_impact": max(0.0, min(1.0, _coerce_float(item.get("score_impact"), default=0.0))),
                "score_actionability": max(
                    0.0, min(1.0, _coerce_float(item.get("score_actionability"), default=0.0))
                ),
                "score_timeliness": max(0.0, min(1.0, _coerce_float(item.get("score_timeliness"), default=0.0))),
                "score_relevance": max(0.0, min(1.0, _coerce_float(item.get("score_relevance"), default=0.0))),
                "reason": reason,
                "summary": summary,
            }
        )
    return result


def _normalize_message(index: int, item: dict[str, Any] | str) -> dict[str, str]:
    if isinstance(item, str):
        return {
            "message_id": f"msg_{index + 1}",
            "chat_id": "",
            "sender_name": "",
            "text": item,
            "create_time": "",
        }
    text = item.get("text")
    if text is None:
        text = item.get("content", "")
    sender = item.get("sender")
    sender_name = ""
    if isinstance(sender, dict):
        sender_name = str(sender.get("name") or sender.get("display_name") or "")
    return {
        "message_id": str(item.get("message_id") or item.get("id") or f"msg_{index + 1}"),
        "chat_id": str(item.get("chat_id") or ""),
        "sender_name": str(item.get("sender_name") or sender_name or ""),
        "text": str(text or ""),
        "create_time": str(item.get("create_time") or item.get("timestamp") or ""),
    }


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


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
