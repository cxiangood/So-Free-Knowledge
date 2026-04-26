#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token filtering with semantic-density/attention-entropy scoring."""

from __future__ import annotations

import re
from statistics import median
from typing import Dict, List

SYMBOL_PATTERN = re.compile(r"^[^\u4e00-\u9fffA-Za-z0-9]+$")
EMPTY_PATTERN = re.compile(r"^\s*$")


def _is_empty_or_symbol(token: str) -> bool:
    return bool(EMPTY_PATTERN.match(token) or SYMBOL_PATTERN.match(token))


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(val - lo) / (hi - lo) for val in values]


def filter_symbols(tokens: List[str]) -> List[str]:
    """Backward-compatible symbol-only filter."""
    return [token for token in tokens if not _is_empty_or_symbol(token)]


def filter_by_semantic_metrics(
    tokens: List[str],
    metrics: Dict[str, Dict[str, float]],
    *,
    alpha: float = 0.6,
    beta: float = 0.4,
    threshold_mode: str = "median",
    remove_symbol_singleton: bool = True,
    return_details: bool = True,
):
    """Filter tokens by semantic metrics first, then remove symbol singletons in meaningful set."""
    vocabulary: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        vocabulary.append(token)

    sd_values: List[float] = []
    ae_values: List[float] = []
    for token in vocabulary:
        rec = metrics.get(token, {})
        sd_values.append(float(rec.get("semantic_density", 0.0)))
        ae_values.append(float(rec.get("attention_entropy", 0.0)))

    sd_norm = _minmax(sd_values)
    ae_norm = _minmax(ae_values)
    scores = [alpha * sd - beta * ae for sd, ae in zip(sd_norm, ae_norm)]

    if threshold_mode != "median":
        raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
    threshold = median(scores) if scores else 0.0

    token_scores: Dict[str, Dict[str, float]] = {}
    meaningful_set = set()
    meaningless_set = set()
    for token, sd, ae, score in zip(vocabulary, sd_values, ae_values, scores):
        token_scores[token] = {
            "semantic_density": sd,
            "attention_entropy": ae,
            "score": score,
        }
        if score >= threshold:
            meaningful_set.add(token)
        else:
            meaningless_set.add(token)

    filtered_tokens: List[str] = []
    for token in tokens:
        if token not in meaningful_set:
            continue
        if remove_symbol_singleton and _is_empty_or_symbol(token):
            continue
        filtered_tokens.append(token)

    if not return_details:
        return filtered_tokens

    return {
        "filtered_tokens": filtered_tokens,
        "meaningful_tokens": sorted(meaningful_set),
        "meaningless_tokens": sorted(meaningless_set),
        "threshold": float(threshold),
        "token_scores": token_scores,
    }

