from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent import SilentKnowledgeAgent, SilentKnowledgeAgentConfig
from ..msg.parse import load_plain_messages_from_archive
from .engine import EngineConfig


DEFAULT_MESSAGES_FILE = Path("message_archive/20260427T043305Z/messages.jsonl")


@dataclass(slots=True)
class OfflineConfig:
    messages_file: str | Path = DEFAULT_MESSAGES_FILE
    output_dir: str | Path = "outputs/local_pipeline"
    state_dir: str | Path = "outputs/local_pipeline/state"
    chat_history_path: str | Path = "outputs/local_pipeline/state/chat_message_store.json"
    chat_history_limit: int = 100
    context_window_size: int = 20
    candidate_threshold: float = 0.45
    knowledge_threshold: float = 0.60
    task_threshold: float = 0.50
    task_push_enabled: bool = False
    task_push_chat_id: str = ""
    env_file: str = ""
    step_trace_enabled: bool = True
    rag_enabled: bool = True
    rag_top_k: int = 5
    rag_min_score: float = 0.35
    rag_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    observe_auto_reply_enabled: bool = True
    observe_ferment_threshold: float = 4.0
    observe_logic1_base: float = 1.0
    observe_logic2_base: float = 1.5
    observe_logic3_base: float = 2.0
    observe_force_non_observe_on_pop: bool = True


def run(config: OfflineConfig | None = None) -> dict[str, Any]:
    cfg = config or OfflineConfig()
    agent = SilentKnowledgeAgent(
        SilentKnowledgeAgentConfig(
            engine=EngineConfig(
                output_dir=cfg.output_dir,
                state_dir=cfg.state_dir,
                chat_history_path=cfg.chat_history_path,
                chat_history_limit=cfg.chat_history_limit,
                context_window_size=cfg.context_window_size,
                candidate_threshold=cfg.candidate_threshold,
                knowledge_threshold=cfg.knowledge_threshold,
                task_threshold=cfg.task_threshold,
                task_push_enabled=cfg.task_push_enabled,
                task_push_chat_id=cfg.task_push_chat_id,
                env_file=cfg.env_file,
                step_trace_enabled=cfg.step_trace_enabled,
                rag_enabled=cfg.rag_enabled,
                rag_top_k=cfg.rag_top_k,
                rag_min_score=cfg.rag_min_score,
                rag_embed_model=cfg.rag_embed_model,
                observe_auto_reply_enabled=cfg.observe_auto_reply_enabled,
                observe_ferment_threshold=cfg.observe_ferment_threshold,
                observe_logic1_base=cfg.observe_logic1_base,
                observe_logic2_base=cfg.observe_logic2_base,
                observe_logic3_base=cfg.observe_logic3_base,
                observe_force_non_observe_on_pop=cfg.observe_force_non_observe_on_pop,
            )
        )
    )

    messages = load_plain_messages_from_archive(cfg.messages_file)
    batch = agent.handle_plain_messages(messages, source="offline_archive", trigger="manual_replay")
    payload = batch.to_dict()

    return {
        **payload,
        "mode": "offline",
        "messages_file": str(cfg.messages_file),
    }


__all__ = ["DEFAULT_MESSAGES_FILE", "OfflineConfig", "run"]
