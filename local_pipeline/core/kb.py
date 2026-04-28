from __future__ import annotations

from ..shared.models import LiftedCard
from ..store.state import LocalStateStore


def save_knowledge(store: LocalStateStore, card: LiftedCard) -> str:
    return store.add_knowledge(card)


__all__ = ["save_knowledge"]
