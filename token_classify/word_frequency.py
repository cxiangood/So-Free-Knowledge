#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Word frequency statistics."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

__all__ = ["summarize_word_frequency"]


def summarize_word_frequency(
    tokens: List[str],
    *,
    top_k: int = 20,
    stop_words: Optional[List[str]] = None,
) -> Dict[str, List]:
    """Build word-frequency ranking and top keywords in one pass."""
    if stop_words:
        stop_words_set = set(stop_words)
        filtered_tokens = [token for token in tokens if token not in stop_words_set]
    else:
        filtered_tokens = list(tokens)

    counter = Counter(filtered_tokens)
    ranked: List[Tuple[str, int]] = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    keywords = [word for word, _ in ranked[: max(0, top_k)]]

    return {
        "word_frequency": ranked,
        "top_keywords": keywords,
    }
