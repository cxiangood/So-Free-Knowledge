from __future__ import annotations

from typing import Any


def normalize_doc(item: dict[str, Any]) -> dict[str, str]:
    from .. import assistant_brief as legacy

    return legacy._normalize_doc(item)


def normalize_message(item: dict[str, Any]) -> dict[str, Any]:
    from .. import assistant_brief as legacy

    return legacy._normalize_message(item)


def normalize_knowledge(item: dict[str, Any]) -> dict[str, str]:
    from .. import assistant_brief as legacy

    return legacy._normalize_knowledge(item)


def normalize_profile(item: dict[str, Any]) -> dict[str, Any]:
    from .. import assistant_brief as legacy

    return legacy._normalize_profile(item)


def normalize_schedule(item: dict[str, Any]) -> dict[str, Any]:
    from .. import assistant_brief as legacy

    return legacy._normalize_schedule(item)


def extract_keywords(text: str) -> set[str]:
    from .. import assistant_brief as legacy

    return legacy._extract_keywords(text)
