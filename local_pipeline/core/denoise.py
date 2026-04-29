from __future__ import annotations

import re

from ..msg.types import MessageEvent

_SPACE_RE = re.compile(r"\s+")
_ONLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_LAUGH_RE = re.compile(r"^(哈|呵|嘿|h|a){4,}$", re.IGNORECASE)
_REPEAT_CHAR_RE = re.compile(r"^(.)\1{4,}$")
_SHORT_ACKS = {
    "ok",
    "okk",
    "k",
    "kk",
    "收到",
    "已阅",
    "好的",
    "嗯",
    "恩",
    "1",
    "2",
    "3",
}


def denoise_messages(messages: list[MessageEvent]) -> tuple[list[MessageEvent], int]:
    kept: list[MessageEvent] = []
    dropped = 0
    for msg in messages:
        text = _normalize(msg.content_text)
        if _is_meaningless(text):
            dropped += 1
            continue
        kept.append(msg)
    return kept, dropped


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "").strip())


def _is_meaningless(text: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    if lower in _SHORT_ACKS:
        return True
    if len(text) <= 1:
        return True
    if _ONLY_PUNCT_RE.match(text):
        return True
    if _REPEAT_CHAR_RE.match(text):
        return True
    if _LAUGH_RE.match(lower):
        return True
    if len(text) <= 3 and not _has_signal_token(text):
        return True
    return False


def _has_signal_token(text: str) -> bool:
    if "?" in text or "？" in text or "!" in text or "！" in text:
        return True
    if any(ch.isdigit() for ch in text):
        return True
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return True
    if "@" in text:
        return True
    return False


__all__ = ["denoise_messages"]
