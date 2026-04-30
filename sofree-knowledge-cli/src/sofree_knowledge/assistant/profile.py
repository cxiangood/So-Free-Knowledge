from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
