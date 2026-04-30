from __future__ import annotations

from typing import Any


def to_markdown(
    documents: list[dict[str, Any]],
    grouped_documents: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    from .. import assistant_brief as legacy

    return legacy._to_markdown(documents, grouped_documents, profile)


def to_card(documents: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    from .. import assistant_brief as legacy

    return legacy._to_card(documents, profile)


def to_interest_card(interest_digest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    from .. import assistant_brief as legacy

    return legacy._to_interest_card(interest_digest, profile)
