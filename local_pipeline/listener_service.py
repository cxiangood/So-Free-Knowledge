from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import lark_oapi as lark

from .message_event_bus import MessageEventBus
from .openapi_message_listener import OpenAPIMessageListener, resolve_listener_credentials


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ListenerServiceConfig:
    env_file: str = ""
    event_types: str = "im.message.receive_v1"
    compact: bool = True
    print_events: bool = True


class ListenerService:
    def __init__(self, config: ListenerServiceConfig) -> None:
        self.config = config
        self.bus = MessageEventBus()
        self._listener: OpenAPIMessageListener | None = None

    def start(self) -> None:
        self._validate_event_types(self.config.event_types)
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


def run_listener_service(args: Any) -> dict[str, Any]:
    config = ListenerServiceConfig(
        env_file=str(getattr(args, "env_file", "") or ""),
        event_types=str(getattr(args, "event_types", "im.message.receive_v1") or "im.message.receive_v1"),
        compact=bool(getattr(args, "compact", True)),
        print_events=bool(getattr(args, "print_events", True)),
    )
    service = ListenerService(config)
    service.start()
    return {
        "ok": True,
        "mode": "listen-messages",
        "event_types": config.event_types,
        "compact": config.compact,
        "print_events": config.print_events,
    }

