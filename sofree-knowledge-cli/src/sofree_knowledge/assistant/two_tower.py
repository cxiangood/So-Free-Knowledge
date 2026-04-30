from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .models import DualTowerConfig


def build_user_tower_text(profile: dict[str, Any]) -> str:
    role = str(profile.get("role") or "").strip()
    persona = str(profile.get("persona") or "").strip()
    interests = [str(item).strip() for item in profile.get("interests", []) if str(item).strip()]
    businesses = [
        str(item.get("name") or "").strip()
        for item in profile.get("business_tracks", [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    parts = [
        f"role: {role}" if role else "",
        f"persona: {persona}" if persona else "",
        f"businesses: {', '.join(businesses)}" if businesses else "",
        f"interests: {', '.join(interests)}" if interests else "",
    ]
    return " | ".join(part for part in parts if part)


def build_document_tower_text(doc: dict[str, Any], *, business: str, doc_type: str) -> str:
    title = str(doc.get("title") or "").strip()
    summary = str(doc.get("summary") or "").strip()
    parts = [
        f"title: {title}" if title else "",
        f"summary: {summary}" if summary else "",
        f"business: {business}" if business else "",
        f"doc_type: {doc_type}" if doc_type else "",
    ]
    return " | ".join(part for part in parts if part)


def build_dual_tower_debug_payload(
    profile: dict[str, Any],
    doc: dict[str, Any],
    *,
    business: str,
    doc_type: str,
    config: DualTowerConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or DualTowerConfig()
    return {
        "enabled": bool(resolved_config.enabled),
        "embedding_model": resolved_config.embedding_model,
        "top_k": int(resolved_config.top_k),
        "min_score": float(resolved_config.min_score),
        "user_tower_text": build_user_tower_text(profile),
        "content_tower_text": build_document_tower_text(doc, business=business, doc_type=doc_type),
    }


def score_dual_tower_texts(user_text: str, content_text: str) -> float:
    user_vec = _text_to_vector(user_text)
    content_vec = _text_to_vector(content_text)
    if not user_vec or not content_vec:
        return 0.0
    numerator = sum(user_vec[token] * content_vec.get(token, 0.0) for token in user_vec)
    user_norm = math.sqrt(sum(value * value for value in user_vec.values()))
    content_norm = math.sqrt(sum(value * value for value in content_vec.values()))
    if user_norm <= 0 or content_norm <= 0:
        return 0.0
    return numerator / (user_norm * content_norm)


def _text_to_vector(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    total = float(sum(counts.values()) or 1.0)
    return {token: count / total for token, count in counts.items()}


def _tokenize(text: str) -> list[str]:
    normalized = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            for idx in range(0, len(token) - 1):
                expanded.append(token[idx : idx + 2])
    return expanded
