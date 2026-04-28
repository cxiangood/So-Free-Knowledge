from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from ..msg.parse import parse_message_event, should_accept_message_event
from ..msg.types import MessageEvent

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
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_listener_credentials(env_file: str = "") -> tuple[str, str]:
    load_listener_env(env_file)
    app_id = (_safe_text(os.getenv("LISTENER_APP_ID")) or _safe_text(os.getenv("SOFREE_FEISHU_APP_ID")) or _safe_text(os.getenv("FEISHU_APP_ID")))
    app_secret = (
        _safe_text(os.getenv("LISTENER_APP_SECRET"))
        or _safe_text(os.getenv("SOFREE_FEISHU_APP_SECRET"))
        or _safe_text(os.getenv("FEISHU_APP_SECRET"))
    )
    if not app_id or not app_secret:
        raise ValueError(
            "Missing listener credentials. Required: LISTENER_APP_ID/SECRET "
            "or SOFREE_FEISHU_APP_ID/SECRET or FEISHU_APP_ID/SECRET."
        )
    return app_id, app_secret


class OpenAPIMessageListener:
    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        bus: MessageEventBus,
        compact: bool = False,
        log_level: lark.LogLevel = lark.LogLevel.INFO,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.bus = bus
        self.compact = compact
        self._event_handler = (
            lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(self._handle_message_receive).build()
        )
        ws_factory = client_factory or lark.ws.Client
        self._client = ws_factory(app_id, app_secret, event_handler=self._event_handler, log_level=log_level)

    def start(self) -> None:
        self._client.start()

    def _handle_message_receive(self, data: Any) -> None:
        event = parse_message_event(data, supported_event_type=SUPPORTED_EVENT_TYPE)
        if event is None:
            return
        if not should_accept_message_event(event):
            return
        dispatch_message_event(self.bus, event)


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
    "resolve_listener_credentials",
]
