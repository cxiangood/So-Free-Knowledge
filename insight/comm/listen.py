from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from utils import getenv_required, load_env_file as _load_env_file

from ..msg.parse import parse_message_event, should_accept_message_event
from ..msg.types import MessageEvent
from .user import get_user_name_by_user_id

try:
    import lark_oapi as lark
except ModuleNotFoundError:  # pragma: no cover
    class _DummyLogLevel:
        INFO = "INFO"

    class _DummyBuilder:
        @staticmethod
        def builder(*args, **kwargs):
            class _HandlerBuilder:
                def register_p2_im_message_receive_v1(self, handler):
                    return self

                def register_p2_im_message_reaction_created_v1(self, handler):
                    return self

                def register_p2_im_message_reaction_deleted_v1(self, handler):
                    return self

                def register_p2_im_message_read_v1(self, handler):
                    return self

                def build(self):
                    return object()

            return _HandlerBuilder()

    class _DummyWS:
        class Client:
            def __init__(self, *args, **kwargs):
                raise ModuleNotFoundError("lark_oapi is required to start OpenAPIMessageListener")

    class _DummyLark:
        LogLevel = _DummyLogLevel
        EventDispatcherHandler = _DummyBuilder
        ws = _DummyWS

    lark = _DummyLark()


LOGGER = logging.getLogger(__name__)
SUPPORTED_EVENT_TYPE = "im.message.receive_v1"
HANDLED_EVENT_TYPES = frozenset({SUPPORTED_EVENT_TYPE})
PERMISSION_COUPLED_IGNORED_EVENT_REGISTER_METHODS = {
    "im.message.reaction.created_v1": "register_p2_im_message_reaction_created_v1",
    "im.message.reaction.deleted_v1": "register_p2_im_message_reaction_deleted_v1",
    "im.message.message_read_v1": "register_p2_im_message_read_v1",
}
SUPPORTED_EVENT_TYPES = frozenset(
    {
        SUPPORTED_EVENT_TYPE,
        *PERMISSION_COUPLED_IGNORED_EVENT_REGISTER_METHODS.keys(),
    }
)
TOPIC_MESSAGE_RECEIVED = "message.received"
TOPIC_MESSAGE_USER_P2P = "message.user.p2p"
TOPIC_MESSAGE_USER_GROUP = "message.user.group"


@dataclass(slots=True)
class PublishedEvent:
    topic: str
    payload: Any


class MessageEventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        normalized = str(topic or "").strip()
        if not normalized:
            raise ValueError("topic is required")
        with self._lock:
            if handler not in self._handlers[normalized]:
                self._handlers[normalized].append(handler)

    def publish(self, topic: str, payload: Any) -> list[PublishedEvent]:
        normalized = str(topic or "").strip()
        if not normalized:
            return []
        with self._lock:
            handlers = list(self._handlers.get(normalized, []))
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                LOGGER.exception("message event handler failed: topic=%s", normalized)
        return [PublishedEvent(topic=normalized, payload=payload)]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def dispatch_message_event(bus: MessageEventBus, event: MessageEvent) -> None:
    bus.publish(TOPIC_MESSAGE_RECEIVED, event)
    if event.chat_type == "p2p":
        bus.publish(TOPIC_MESSAGE_USER_P2P, event)
    elif event.chat_type == "group":
        bus.publish(TOPIC_MESSAGE_USER_GROUP, event)


def load_listener_env(path: str = "") -> None:
    env_path = Path(path).expanduser() if path.strip() else Path.cwd() / ".env"
    _load_env_file(env_path)


def resolve_listener_credentials(env_file: str = "") -> tuple[str, str]:
    load_listener_env(env_file)
    return _safe_text(getenv_required("APP_ID")), _safe_text(getenv_required("SECRET_ID"))


def parse_event_types(event_types: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    if event_types is None:
        return {SUPPORTED_EVENT_TYPE}
    if isinstance(event_types, str):
        values = event_types.split(",")
    else:
        values = list(event_types)
    normalized = {str(value or "").strip() for value in values}
    normalized = {value for value in normalized if value}
    return normalized or {SUPPORTED_EVENT_TYPE}


class OpenAPIMessageListener:
    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        bus: MessageEventBus,
        compact: bool = False,
        event_types: str | list[str] | tuple[str, ...] | set[str] | None = None,
        log_level: lark.LogLevel = lark.LogLevel.INFO,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.bus = bus
        self.compact = compact
        self._app_id = app_id
        self._app_secret = app_secret
        self._enabled_event_types = parse_event_types(event_types)
        self._event_handler = self._build_event_handler()
        ws_factory = client_factory or lark.ws.Client
        self._client = ws_factory(app_id, app_secret, event_handler=self._event_handler, log_level=log_level)

    def start(self) -> None:
        self._client.start()

    def _build_event_handler(self) -> Any:
        builder = lark.EventDispatcherHandler.builder("", "")
        unknown_event_types = self._enabled_event_types - SUPPORTED_EVENT_TYPES
        if unknown_event_types:
            LOGGER.warning("unsupported event_types ignored: %s", sorted(unknown_event_types))

        if SUPPORTED_EVENT_TYPE in self._enabled_event_types:
            builder = builder.register_p2_im_message_receive_v1(self._handle_message_receive)
        else:
            builder = builder.register_p2_im_message_receive_v1(self._handle_ignored_event)

        ignored_enabled_event_types = self._enabled_event_types - HANDLED_EVENT_TYPES - unknown_event_types
        if ignored_enabled_event_types:
            LOGGER.warning(
                "event_types have SDK processors but no insight pipeline handler yet: %s",
                sorted(ignored_enabled_event_types),
            )

        # These events are bound to the same Lark permission set as message
        # listening, so the app may receive them even when insight only handles
        # messages. Register no-op processors to avoid Lark SDK ERROR logs.
        for event_type, method_name in PERMISSION_COUPLED_IGNORED_EVENT_REGISTER_METHODS.items():
            register = getattr(builder, method_name, None)
            if register is None:
                LOGGER.warning("lark sdk processor register method not found: event_type=%s method=%s", event_type, method_name)
                continue
            builder = register(self._handle_ignored_event)

        return builder.build()

    def _handle_message_receive(self, data: Any) -> None:
        event = parse_message_event(
            data,
            supported_event_type=SUPPORTED_EVENT_TYPE,
            user_name_resolver=self._resolve_user_name,
        )
        if event is None:
            return
        if not should_accept_message_event(event):
            return
        dispatch_message_event(self.bus, event)

    def _resolve_user_name(self, user_id: str) -> str:
        try:
            return get_user_name_by_user_id(user_id, self._app_id, self._app_secret)
        except Exception:
            LOGGER.debug("resolve user name failed for user_id=%s", user_id, exc_info=True)
            return ""

    def _handle_ignored_event(self, data: Any) -> None:
        event_type = _event_type_from_lark_payload(data)
        LOGGER.warning("ignored lark event without insight pipeline handler: event_type=%s", event_type or "unknown")


def _event_type_from_lark_payload(data: Any) -> str:
    header = getattr(data, "header", None)
    event_type = getattr(header, "event_type", "") or getattr(data, "event_type", "")
    if event_type:
        return str(event_type)
    event = getattr(data, "event", None)
    event_type = getattr(event, "event_type", "")
    return str(event_type or "")


__all__ = [
    "MessageEvent",
    "MessageEventBus",
    "OpenAPIMessageListener",
    "TOPIC_MESSAGE_RECEIVED",
    "TOPIC_MESSAGE_USER_P2P",
    "TOPIC_MESSAGE_USER_GROUP",
    "parse_message_event",
    "should_accept_message_event",
    "dispatch_message_event",
    "parse_event_types",
    "resolve_listener_credentials",
]
