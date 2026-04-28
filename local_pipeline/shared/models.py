from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .utils import now_utc_iso


RouteTarget = Literal["knowledge", "task", "observe"]


@dataclass(slots=True)
class InspirationCandidate:
    candidate_id: str
    source_message_ids: list[str]
    score_total: float
    score_breakdown: dict[str, float]
    reasons: list[str]
    evidence: str
    content: str
    created_at: str = field(default_factory=now_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LiftedCard:
    card_id: str
    candidate_id: str
    title: str
    summary: str
    problem: str
    suggestion: str
    target_audience: str
    evidence: list[str]
    tags: list[str]
    confidence: float
    suggested_target: RouteTarget
    source_message_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouteDecision:
    card_id: str
    target_pool: RouteTarget
    reason_codes: list[str]
    threshold_snapshot: dict[str, float]
    stored_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PushEvent:
    event_id: str
    trigger_type: str
    target: str
    payload_ref: str
    task_id: str = ""
    delivery_status: str = "not_applicable"
    message_id: str = ""
    error: str = ""
    created_at: str = field(default_factory=now_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["RouteTarget", "InspirationCandidate", "LiftedCard", "RouteDecision", "PushEvent"]
