#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token filtering: only remove invalid symbol-only tokens."""

from __future__ import annotations

import re
from typing import List

SYMBOL_PATTERN = re.compile(r"^[^\u4e00-\u9fffA-Za-z0-9]+$")
EMPTY_PATTERN = re.compile(r"^\s*$")

__all__ = ["filter_invalid_tokens"]


def _is_invalid_token(token: str) -> bool:
    return bool(EMPTY_PATTERN.match(token) or SYMBOL_PATTERN.match(token))


def filter_invalid_tokens(tokens: List[str]) -> List[str]:
    """Keep only valid tokens; invalid means empty or symbol-only."""
    return [token for token in tokens if not _is_invalid_token(token)]

