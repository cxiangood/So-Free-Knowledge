from __future__ import annotations

from .index import VectorKnowledgeStore


def retrieve(store: VectorKnowledgeStore | None, *, query: str, top_k: int, min_score: float) -> list:
    if store is None:
        return []
    return store.search(query=query, top_k=top_k, min_score=min_score)
