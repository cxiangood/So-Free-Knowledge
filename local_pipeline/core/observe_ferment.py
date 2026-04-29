from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..msg.types import MessageEvent
from ..shared.models import LiftedCard, ObserveFermentResult
from ..store.state import LocalStateStore

RISK_TERMS = ("故障", "失败", "超时", "阻塞", "风险", "异常", "报警", "回滚")
_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+")


@dataclass(slots=True)
class ObserveMatch:
    observe_id: str
    score: float


def apply_logic1_on_observe_add(
    *,
    card: LiftedCard,
    message: MessageEvent,
    state_store: LocalStateStore,
    threshold: float,
    base_score: float,
) -> ObserveFermentResult | None:
    observe_id = _observe_id_from_card(state_store, card)
    if not observe_id:
        return None
    bonus = 0.0
    if "?" in message.content_text or "？" in message.content_text:
        bonus += 0.5
    if len(message.mentions) >= 2:
        bonus += 0.5
    if any(term in message.content_text for term in RISK_TERMS):
        bonus += 0.5
    added = float(base_score) + min(0.5, bonus)
    row = state_store.apply_observe_ferment(observe_id, logic="logic1", score_added=added)
    if row is None:
        return None
    ferment_score = float(row.get("ferment_score", 0.0) or 0.0)
    return ObserveFermentResult(
        observe_id=observe_id,
        logic="logic1",
        score_added=round(added, 4),
        ferment_score=ferment_score,
        triggered_pop=ferment_score >= float(threshold),
    )


def apply_logic2_on_knowledge(*, card: LiftedCard, state_store: LocalStateStore, threshold: float, base_score: float) -> list[ObserveFermentResult]:
    return _apply_logic_on_similar(card=card, logic="logic2", state_store=state_store, threshold=threshold, base_score=base_score)


def apply_logic3_on_task(*, card: LiftedCard, state_store: LocalStateStore, threshold: float, base_score: float) -> list[ObserveFermentResult]:
    return _apply_logic_on_similar(card=card, logic="logic3", state_store=state_store, threshold=threshold, base_score=base_score)


def pop_ready_items(state_store: LocalStateStore, *, threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in state_store.list_observe_items():
        if str(item.get("pop_status", "tracking")) != "tracking":
            continue
        if float(item.get("ferment_score", 0.0) or 0.0) >= float(threshold):
            out.append(item)
    return out


def _apply_logic_on_similar(
    *,
    card: LiftedCard,
    logic: str,
    state_store: LocalStateStore,
    threshold: float,
    base_score: float,
) -> list[ObserveFermentResult]:
    results: list[ObserveFermentResult] = []
    for item in state_store.list_observe_items():
        if str(item.get("pop_status", "tracking")) != "tracking":
            continue
        matched = _similarity_score(card, item)
        if matched < 0.35:
            continue
        observe_id = str(item.get("observe_id", ""))
        row = state_store.apply_observe_ferment(observe_id, logic=logic, score_added=float(base_score))
        if row is None:
            continue
        ferment_score = float(row.get("ferment_score", 0.0) or 0.0)
        results.append(
            ObserveFermentResult(
                observe_id=observe_id,
                logic=logic,
                score_added=float(base_score),
                ferment_score=ferment_score,
                triggered_pop=ferment_score >= float(threshold),
            )
        )
    return results


def _observe_id_from_card(state_store: LocalStateStore, card: LiftedCard) -> str:
    topic = card.title.strip().lower()
    for item in state_store.list_observe_items():
        if str(item.get("topic", "")).strip().lower() == topic:
            return str(item.get("observe_id", ""))
    return ""


def _tokens(text: str) -> set[str]:
    return {item.lower() for item in _WORD_RE.findall(str(text or "")) if len(item.strip()) >= 2}


def _similarity_score(card: LiftedCard, item: dict[str, Any]) -> float:
    left = _tokens(" ".join([card.title, card.summary] + card.evidence[:2]))
    right = _tokens(" ".join([str(item.get("topic", ""))] + [str(v) for v in item.get("evidence", [])[:3]]))
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / max(1, union)


__all__ = [
    "ObserveFermentResult",
    "ObserveMatch",
    "apply_logic1_on_observe_add",
    "apply_logic2_on_knowledge",
    "apply_logic3_on_task",
    "pop_ready_items",
]
