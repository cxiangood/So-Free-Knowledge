from __future__ import annotations

import atexit
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from utils import get_config_bool, get_config_float, get_config_int, get_config_str

from ..comm.listen import MessageEventBus, OpenAPIMessageListener, TOPIC_MESSAGE_RECEIVED, resolve_listener_credentials
from .engine import Engine, EngineConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OnlineConfig:
    env_file: str = get_config_str("insight.env_file")
    event_types: str = get_config_str("insight.event_types")
    compact: bool = get_config_bool("insight.compact")
    output_dir: str = get_config_str("insight.output_dir")
    state_dir: str = get_config_str("insight.state_dir")
    chat_history_path: str = get_config_str("insight.chat_history_path")
    chat_history_limit: int = get_config_int("insight.chat_history_limit")
    context_window_size: int = get_config_int("insight.context_window_size")
    detect_threshold: float = get_config_float("insight.detect_threshold")
    task_push_enabled: bool = get_config_bool("insight.task_push_enabled")
    step_trace_enabled: bool = get_config_bool("insight.step_trace_enabled")
    rag_enabled: bool = get_config_bool("insight.rag_enabled")
    rag_top_k: int = get_config_int("insight.rag_top_k")
    rag_min_score: float = get_config_float("insight.rag_min_score")
    rag_embed_model: str = get_config_str("insight.rag_embed_model")
    observe_auto_reply_enabled: bool = get_config_bool("insight.observe_auto_reply_enabled")
    observe_ferment_threshold: float = get_config_float("insight.observe_ferment_threshold")
    observe_logic1_base: float = get_config_float("insight.observe_logic1_base")
    observe_logic2_base: float = get_config_float("insight.observe_logic2_base")
    observe_logic3_base: float = get_config_float("insight.observe_logic3_base")
    observe_force_non_observe_on_pop: bool = get_config_bool("insight.observe_force_non_observe_on_pop")
    max_workers: int = get_config_int("insight.online.max_workers")


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
    worker_count = max(1, int(cfg.max_workers or 1))
    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="msg-worker")

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
        event_types=cfg.event_types,
    )
    LOGGER.info("Online pipeline started with async message processing (%s workers)", worker_count)
    listener.start()


__all__ = ["OnlineConfig", "start"]
