from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROFILE_STOPWORDS = {
    "今天",
    "今晚",
    "明天",
    "相关",
    "检查",
    "确认",
    "一起",
    "需要",
    "进行",
    "安排",
    "项目",
    "文档",
    "消息",
    "讨论",
    "功能",
    "内容",
    "处理",
    "更新",
    "please",
    "check",
    "today",
    "tonight",
    "review",
}


def assistant_profile_default_path(output_dir: str) -> Path:
    return Path(output_dir).expanduser() / "assistant_profile.json"


def load_assistant_profile_config(
    *,
    output_dir: str,
    profile_file: str = "",
) -> dict[str, Any]:
    path = Path(profile_file).expanduser() if str(profile_file or "").strip() else assistant_profile_default_path(output_dir)
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_profile_overrides(
    *,
    persona: str = "",
    role: str = "",
    businesses: str = "",
    interests: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if str(persona or "").strip():
        payload["persona"] = str(persona).strip()
    if str(role or "").strip():
        payload["role"] = str(role).strip()
    if str(businesses or "").strip():
        payload["businesses"] = [item.strip() for item in str(businesses).split(",") if item.strip()]
    if str(interests or "").strip():
        payload["interests"] = [item.strip() for item in str(interests).split(",") if item.strip()]
    return payload


def build_schedule_overrides(
    *,
    mode: str = "",
    timezone: str = "",
    weekly_brief_cron: str = "",
    nightly_interest_cron: str = "",
    weekly_enabled: bool | None = None,
    nightly_enabled: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if str(mode or "").strip():
        payload["mode"] = str(mode).strip()
    if str(timezone or "").strip():
        payload["timezone"] = str(timezone).strip()
    if str(weekly_brief_cron or "").strip():
        payload["weekly_brief_cron"] = str(weekly_brief_cron).strip()
    if str(nightly_interest_cron or "").strip():
        payload["nightly_interest_cron"] = str(nightly_interest_cron).strip()
    if weekly_enabled is not None:
        payload["weekly_enabled"] = bool(weekly_enabled)
    if nightly_enabled is not None:
        payload["nightly_enabled"] = bool(nightly_enabled)
    return payload


def build_retrieval_overrides(
    *,
    dual_tower_enabled: bool | None = None,
    dual_tower_model: str = "",
    dual_tower_model_file: str = "",
    dual_tower_top_k: int | None = None,
    dual_tower_min_score: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if dual_tower_enabled is not None:
        payload["enabled"] = bool(dual_tower_enabled)
    if str(dual_tower_model or "").strip():
        payload["embedding_model"] = str(dual_tower_model).strip()
    if str(dual_tower_model_file or "").strip():
        payload["model_file"] = str(dual_tower_model_file).strip()
    if dual_tower_top_k is not None:
        payload["top_k"] = max(1, int(dual_tower_top_k))
    if dual_tower_min_score is not None:
        payload["min_score"] = max(0.0, float(dual_tower_min_score))
    return payload


def save_assistant_profile_config(
    *,
    output_dir: str,
    profile_file: str = "",
    payload: dict[str, Any],
) -> Path:
    path = Path(profile_file).expanduser() if str(profile_file or "").strip() else assistant_profile_default_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def suggest_profile_from_online_inputs(
    *,
    online_inputs: dict[str, Any],
    display_name: str = "",
    existing_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = dict(existing_profile or {})
    interests = list(existing.get("interests") or [])
    businesses = list(existing.get("businesses") or [])
    if not interests:
        interests = _suggest_interest_terms(online_inputs)[:4]
    if not businesses:
        businesses = interests[:2]
    role = str(existing.get("role") or "").strip() or "待确认角色"
    persona = str(existing.get("persona") or "").strip()
    if not persona:
        focus = "、".join(interests[:2]) if interests else "协同推进"
        prefix = f"{display_name}：" if display_name else ""
        persona = f"{prefix}关注{focus}的务实推进型"
    profile = {
        "display_name": display_name,
        "role": role,
        "persona": persona,
        "businesses": businesses,
        "interests": interests,
        "require_user_confirmation": True,
    }
    return profile


def build_profile_review_card(
    *,
    profile: dict[str, Any],
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    businesses = "、".join(profile.get("businesses", [])[:3]) or "待确认"
    interests = "、".join(profile.get("interests", [])[:4]) or "待确认"
    source = source_meta or {}
    source_line = (
        f"基于最近 {int(source.get('message_count', 0))} 条消息、"
        f"{int(source.get('document_count', 0))} 篇文档生成"
    )
    content = [
        f"- 角色：{profile.get('role') or '待确认'}",
        f"- 形象：{profile.get('persona') or '待确认'}",
        f"- 关注业务：{businesses}",
        f"- 兴趣词：{interests}",
        "",
        f"> {source_line}",
        "> 你可以直接同意当前画像，也可以运行 `assistant set-profile` 修改。",
    ]
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "wathet",
            "title": {"tag": "plain_text", "content": "AI 画像初始化建议"},
            "subtitle": {"tag": "plain_text", "content": "授权完成后请先确认画像"},
        },
        "elements": [{"tag": "markdown", "content": "\n".join(content)}],
    }


def _suggest_interest_terms(online_inputs: dict[str, Any]) -> list[str]:
    counter: Counter[str] = Counter()
    for item in online_inputs.get("knowledge_items", []) or []:
        title = str(item.get("title") or "").strip()
        if title:
            counter[title] += 3
    for doc in online_inputs.get("documents", []) or []:
        for token in _extract_profile_terms(" ".join([str(doc.get("title") or ""), str(doc.get("summary") or "")])):
            counter[token] += 2
    for message in online_inputs.get("messages", []) or []:
        for token in _extract_profile_terms(str(message.get("text") or message.get("content") or "")):
            counter[token] += 1
    return [token for token, _ in counter.most_common(8)]


def _extract_profile_terms(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,6}", str(text or ""))
    results: list[str] = []
    for token in tokens:
        normalized = token.strip()
        if not normalized:
            continue
        if normalized.lower() in PROFILE_STOPWORDS:
            continue
        results.append(normalized)
    return results
