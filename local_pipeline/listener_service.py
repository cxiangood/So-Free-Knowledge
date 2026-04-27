from __future__ import annotations

import logging
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
from .openapi_message_listener import OpenAPIMessageListener, resolve_listener_credentials


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ListenerServiceConfig:
    env_file: str = ""
    event_types: str = "im.message.receive_v1"
    compact: bool = True
    print_events: bool = True
    chat_history_path: str = "outputs/local_pipeline/state/chat_message_store.json"
    chat_history_limit: int = 100


class ListenerService:
    def __init__(self, config: ListenerServiceConfig) -> None:
        self.config = config
        self.bus = MessageEventBus()
        self._listener: OpenAPIMessageListener | None = None
        self._chat_store: ChatMessageStore | None = None

    def start(self) -> None:
        self._validate_event_types(self.config.event_types)
        self._chat_store = ChatMessageStore(
            self.config.chat_history_path,
            max_messages_per_chat=self.config.chat_history_limit,
        )
        self.bus.subscribe(TOPIC_MESSAGE_RECEIVED, self._record_message)
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
            self._chat_store.append(event)
        except Exception:
            LOGGER.exception("failed to record chat message")

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
