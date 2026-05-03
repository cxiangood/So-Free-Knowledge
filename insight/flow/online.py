from __future__ import annotations

import atexit
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..comm.listen import MessageEventBus, OpenAPIMessageListener, TOPIC_MESSAGE_RECEIVED, resolve_listener_credentials
from .engine import Engine, EngineConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OnlineConfig:
    env_file: str = ""
    event_types: str = "im.message.receive_v1"
    compact: bool = False
    output_dir: str = "outputs/local_pipeline"
    state_dir: str = "outputs/local_pipeline/state"
    chat_history_path: str = "outputs/local_pipeline/state/chat_message_store.json"
    chat_history_limit: int = 100
    context_window_size: int = 20
    detect_threshold: float = 45.0
    task_push_enabled: bool = False
    task_push_chat_id: str = ""
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
            detect_threshold=cfg.detect_threshold,
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

    # 初始化线程池实现异步消息处理，避免阻塞WebSocket线程
    executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="msg-worker")

    # 注册进程退出钩子，平滑关闭线程池，等待任务完成
    def _shutdown_executor():
        LOGGER.info("Shutting down message worker thread pool, waiting for pending tasks...")
        executor.shutdown(wait=True, cancel_futures=False)
        LOGGER.info("Message worker thread pool shutdown complete.")
    atexit.register(_shutdown_executor)

    # 异步消息处理函数
    def async_handle_message(evt):
        try:
            engine.run(evt, context={"mode": "online"})
        except Exception as e:
            LOGGER.exception("Async message handling failed: message_id=%s, chat_id=%s",
                           evt.message_id, evt.chat_id)

    # 订阅消息，提交到线程池异步执行
    bus.subscribe(TOPIC_MESSAGE_RECEIVED, lambda evt: executor.submit(async_handle_message, evt))

    app_id, app_secret = resolve_listener_credentials(cfg.env_file)
    listener = OpenAPIMessageListener(
        app_id=app_id,
        app_secret=app_secret,
        bus=bus,
        compact=cfg.compact,
    )
    LOGGER.info("Online pipeline started with async message processing (8 workers)")
    listener.start()


__all__ = ["OnlineConfig", "start"]
