from __future__ import annotations

from ..shared.models import LiftedCard
from ..store.state import LocalStateStore


def save_observe(store: LocalStateStore, card: LiftedCard) -> str:
    return store.add_observe(card)


__all__ = ["save_observe"]
