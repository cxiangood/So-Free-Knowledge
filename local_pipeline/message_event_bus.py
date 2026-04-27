from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any


LOGGER = logging.getLogger(__name__)

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

    def unsubscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        normalized = str(topic or "").strip()
        if not normalized:
            return
        with self._lock:
            handlers = self._handlers.get(normalized, [])
            if handler in handlers:
                handlers.remove(handler)
            if not handlers and normalized in self._handlers:
                self._handlers.pop(normalized, None)

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

