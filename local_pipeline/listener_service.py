from __future__ import annotations

import logging
import json
from dataclasses import dataclass

try:
    import lark_oapi as lark
except ModuleNotFoundError:  # pragma: no cover - fallback for local tests without SDK
    class _DummyLogLevel:
        INFO = "INFO"

    class _DummyLark:
        LogLevel = _DummyLogLevel

    lark = _DummyLark()

from .chat_message_store import ChatMessageStore
from .message_event_bus import MessageEventBus, TOPIC_MESSAGE_RECEIVED
from .openapi_message_listener import MessageEvent, OpenAPIMessageListener, resolve_listener_credentials
from .realtime_processor import RealtimeProcessor, RealtimeProcessorConfig


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ListenerServiceConfig:
    env_file: str = ""
    event_types: str = "im.message.receive_v1"
    compact: bool = False
    print_events: bool = True
    output_dir: str = "outputs/local_pipeline"
    chat_history_path: str = "outputs/local_pipeline/state/chat_message_store.json"
    chat_history_limit: int = 100
    state_dir: str = "outputs/local_pipeline/state"
    enable_llm: bool = False
    context_window_size: int = 20
    task_push_enabled: bool = False
    task_push_chat_id: str = ""
    candidate_threshold: float = 0.45
    knowledge_threshold: float = 0.60
    task_threshold: float = 0.50
    step_trace_enabled: bool = True


class ListenerService:
    def __init__(self, config: ListenerServiceConfig) -> None:
        self.config = config
        self.bus = MessageEventBus()
        self._listener: OpenAPIMessageListener | None = None
        self._chat_store: ChatMessageStore | None = None
        self._processor: RealtimeProcessor | None = None

    def start(self) -> None:
        self._validate_event_types(self.config.event_types)
        self._chat_store = ChatMessageStore(
            self.config.chat_history_path,
            max_messages_per_chat=self.config.chat_history_limit,
        )
        self.bus.subscribe(TOPIC_MESSAGE_RECEIVED, self._record_message)
        self._processor = RealtimeProcessor(
            chat_store=self._chat_store,
            config=RealtimeProcessorConfig(
                state_dir=self.config.state_dir,
                output_dir=self.config.output_dir,
                context_window_size=self.config.context_window_size,
                enable_llm=self.config.enable_llm,
                candidate_threshold=self.config.candidate_threshold,
                knowledge_threshold=self.config.knowledge_threshold,
                task_threshold=self.config.task_threshold,
                task_push_enabled=self.config.task_push_enabled,
                task_push_chat_id=self.config.task_push_chat_id,
                env_file=self.config.env_file,
                step_trace_enabled=self.config.step_trace_enabled,
            ),
        )
        self.bus.subscribe(TOPIC_MESSAGE_RECEIVED, self._process_message)
        app_id, app_secret = resolve_listener_credentials(self.config.env_file)
        self._listener = OpenAPIMessageListener(
            app_id=app_id,
            app_secret=app_secret,
            bus=self.bus,
            compact=bool(self.config.compact),
            print_events=bool(self.config.print_events),
            log_level=lark.LogLevel.INFO,
        )
        LOGGER.info("listener service started with event_types=%s", self.config.event_types)
        self._listener.start()

    def _record_message(self, event: object) -> None:
        if self._chat_store is None:
            return
        try:
            cached = self._chat_store.append(event)
            if bool(self.config.step_trace_enabled):
                payload = {
                    "type": "realtime_step",
                    "message_id": getattr(event, "message_id", ""),
                    "chat_id": getattr(event, "chat_id", ""),
                    "step": "cached",
                    "status": "ok" if cached else "skipped",
                    "detail": "chat_message_store updated" if cached else "cache skipped",
                    "input": {
                        "event": event.to_dict() if hasattr(event, "to_dict") and callable(getattr(event, "to_dict")) else "",
                        "chat_history_path": self.config.chat_history_path,
                        "chat_history_limit": self.config.chat_history_limit,
                    },
                    "output": {
                        "cached": bool(cached),
                        "cached_event": cached if isinstance(cached, dict) else {},
                    },
                }
                line = json.dumps(payload, ensure_ascii=False)
                LOGGER.info(line)
                print(line, flush=True)
        except Exception:
            LOGGER.exception("failed to record chat message")

    def _process_message(self, event: object) -> None:
        if self._processor is None:
            return
        if not isinstance(event, MessageEvent):
            return
        try:
            self._processor.process_incoming_event(event)
        except Exception:
            LOGGER.exception("failed to process realtime message")

    @staticmethod
    def _validate_event_types(event_types: str) -> None:
        values = [item.strip() for item in str(event_types or "").split(",") if item.strip()]
        if not values:
            return
        if "im.message.receive_v1" not in values:
            raise ValueError("This listener only supports event type: im.message.receive_v1")
        for value in values:
            if value != "im.message.receive_v1":
                LOGGER.warning("unsupported event type ignored: %s", value)
