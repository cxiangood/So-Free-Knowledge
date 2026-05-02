from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..flow.engine import EngineConfig
from ..msg.cache import ChatMessageStore
from ..rag.index import VectorKnowledgeStore
from ..rag.retriever import retrieve
from ..store.state import LocalStateStore


@dataclass(slots=True)
class AgentMemoryConfig:
    state_dir: str | Path
    chat_history_path: str | Path
    chat_history_limit: int
    rag_enabled: bool
    rag_embed_model: str
    rag_top_k: int
    rag_min_score: float

    @classmethod
    def from_engine_config(cls, config: EngineConfig) -> "AgentMemoryConfig":
        return cls(
            state_dir=config.state_dir,
            chat_history_path=config.chat_history_path,
            chat_history_limit=config.chat_history_limit,
            rag_enabled=config.rag_enabled,
            rag_embed_model=config.rag_embed_model,
            rag_top_k=config.rag_top_k,
            rag_min_score=config.rag_min_score,
        )


class AgentMemory:
    def __init__(self, config: AgentMemoryConfig) -> None:
        self.config = config
        self.state_store = LocalStateStore(config.state_dir)
        self.chat_store = ChatMessageStore(path=config.chat_history_path, max_messages_per_chat=config.chat_history_limit)
        self.vector_store: VectorKnowledgeStore | None = None
        if config.rag_enabled:
            try:
                self.vector_store = VectorKnowledgeStore(Path(config.state_dir) / "vector_kb", embed_model=config.rag_embed_model)
            except Exception:
                self.vector_store = None

    def recent_chat_messages(self, chat_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.chat_store.get_chat_messages(chat_id)
        return rows[-max(1, int(limit or 1)) :]

    def search_knowledge(self, query: str, *, top_k: int | None = None, min_score: float | None = None) -> list:
        return retrieve(
            self.vector_store,
            query=query,
            top_k=self.config.rag_top_k if top_k is None else top_k,
            min_score=self.config.rag_min_score if min_score is None else min_score,
        )

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return self.state_store.snapshot()


__all__ = ["AgentMemory", "AgentMemoryConfig"]
