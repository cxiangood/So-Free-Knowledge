from __future__ import annotations

from ..rag.index import VectorKnowledgeStore
from ..shared.models import LiftedCard
from ..store.state import LocalStateStore


def save_knowledge(store: LocalStateStore, card: LiftedCard, vector_store: VectorKnowledgeStore | None = None) -> str:
    knowledge_id = store.add_knowledge(card)
    if vector_store is not None:
        try:
            vector_store.upsert_knowledge(
                knowledge_id=knowledge_id,
                card_id=card.card_id,
                title=card.title,
                topic_focus=card.topic_focus,
                summary=card.summary,
                times=card.times,
                locations=card.locations,
                participants=card.participants,
            )
        except Exception:
            pass
    return knowledge_id


__all__ = ["save_knowledge"]
