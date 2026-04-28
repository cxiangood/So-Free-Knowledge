from __future__ import annotations

from dataclasses import dataclass

from ..comm.listen import MessageEventBus, OpenAPIMessageListener, TOPIC_MESSAGE_RECEIVED, resolve_listener_credentials
from .engine import Engine, EngineConfig


@dataclass(slots=True)
class OnlineConfig:
    env_file: str = ""
    event_types: str = "im.message.receive_v1"
    compact: bool = False
    output_dir: str = "outputs/local_pipeline"
    state_dir: str = "outputs/local_pipeline/state"
    chat_history_path: str = "outputs/local_pipeline/state/chat_message_store.json"
    chat_history_limit: int = 100
    enable_llm: bool = False
    context_window_size: int = 20
    task_push_enabled: bool = False
    task_push_chat_id: str = ""
    candidate_threshold: float = 0.45
    knowledge_threshold: float = 0.60
    task_threshold: float = 0.50
    step_trace_enabled: bool = True
    rag_enabled: bool = True
    rag_top_k: int = 5
    rag_min_score: float = 0.35
    rag_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    observe_auto_reply_enabled: bool = True


def start(config: OnlineConfig | None = None) -> None:
    cfg = config or OnlineConfig()
    bus = MessageEventBus()
    engine = Engine(
        EngineConfig(
            output_dir=cfg.output_dir,
            state_dir=cfg.state_dir,
            chat_history_path=cfg.chat_history_path,
            chat_history_limit=cfg.chat_history_limit,
            context_window_size=cfg.context_window_size,
            enable_llm=cfg.enable_llm,
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
        )
    )
    bus.subscribe(TOPIC_MESSAGE_RECEIVED, lambda evt: engine.run(evt, context={"mode": "online"}))
    app_id, app_secret = resolve_listener_credentials(cfg.env_file)
    listener = OpenAPIMessageListener(
        app_id=app_id,
        app_secret=app_secret,
        bus=bus,
        compact=cfg.compact,
    )
    listener.start()


__all__ = ["OnlineConfig", "start"]
