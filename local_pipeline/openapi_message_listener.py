from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import lark_oapi as lark
except ModuleNotFoundError:  # pragma: no cover - fallback for local tests without SDK
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

from .message_event_bus import (
    MessageEventBus,
    TOPIC_MESSAGE_RECEIVED,
    TOPIC_MESSAGE_USER_GROUP,
    TOPIC_MESSAGE_USER_P2P,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_EVENT_TYPE = "im.message.receive_v1"


@dataclass(slots=True)
class MessageEvent:
    event_type: str
    event_id: str
    create_time: str
    message_id: str
    chat_id: str
    chat_type: str
    message_type: str
    content_text: str
    content_raw: str
    root_id: str
    parent_id: str
    update_time: str
    thread_id: str
    sender_open_id: str
    sender_union_id: str
    sender_user_id: str
    sender_type: str
    tenant_key: str
    mentions: list[dict[str, Any]] = field(default_factory=list)
    user_agent: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.raw, dict) and isinstance(self.raw.get("event"), dict):
            return {"event": self.raw["event"]}
        return {
            "event": {
                "sender": {
                    "sender_id": {
                        "union_id": self.sender_union_id,
                        "user_id": self.sender_user_id,
                        "open_id": self.sender_open_id,
                    },
                    "sender_type": self.sender_type,
                    "tenant_key": self.tenant_key,
                },
                "message": {
                    "message_id": self.message_id,
                    "root_id": self.root_id,
                    "parent_id": self.parent_id,
                    "create_time": self.create_time,
                    "update_time": self.update_time,
                    "chat_id": self.chat_id,
                    "thread_id": self.thread_id,
                    "chat_type": self.chat_type,
                    "message_type": self.message_type,
                    "content": self.content_raw,
                    "mentions": self.mentions,
                    "user_agent": self.user_agent,
                },
            }
        }

    def to_compact_dict(self) -> dict[str, Any]:
        return self.to_dict()


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _get_field(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _sender_id_dict(sender_id: Any) -> dict[str, str]:
    return {
        "union_id": _safe_text(_get_field(sender_id, "union_id", "")),
        "user_id": _safe_text(_get_field(sender_id, "user_id", "")),
        "open_id": _safe_text(_get_field(sender_id, "open_id", "")),
    }


def _mentions_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            mention_id = _sender_id_dict(item.get("id", {}))
            rows.append(
                {
                    "key": _safe_text(item.get("key", "")),
                    "id": mention_id,
                    "mentioned_type": _safe_text(item.get("mentioned_type", "")),
                    "name": _safe_text(item.get("name", "")),
                    "tenant_key": _safe_text(item.get("tenant_key", "")),
                }
            )
            continue
        mention_id = _sender_id_dict(_get_field(item, "id", None))
        rows.append(
            {
                "key": _safe_text(_get_field(item, "key", "")),
                "id": mention_id,
                "mentioned_type": _safe_text(_get_field(item, "mentioned_type", "")),
                "name": _safe_text(_get_field(item, "name", "")),
                "tenant_key": _safe_text(_get_field(item, "tenant_key", "")),
            }
        )
    return rows


def _parse_content_text(raw_content: Any) -> str:
    content = _safe_text(raw_content)
    if not content:
        return ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    if not isinstance(payload, dict):
        return content

    text_value = payload.get("text")
    if isinstance(text_value, str):
        return text_value

    # post content
    post = payload.get("content")
    if isinstance(post, list):
        lines: list[str] = []
        for row in post:
            if not isinstance(row, list):
                continue
            segs: list[str] = []
            for item in row:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    segs.append(text)
            joined = "".join(segs).strip()
            if joined:
                lines.append(joined)
        if lines:
            return "\n".join(lines)

    for key in ("title", "template"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return content


def parse_message_event(data: Any) -> MessageEvent | None:
    header = _get_field(data, "header", None)
    event = _get_field(data, "event", None)
    if event is None:
        return None
    message = _get_field(event, "message", None)
    sender = _get_field(event, "sender", None)
    if message is None or sender is None:
        return None

    sender_id = _get_field(sender, "sender_id", None)
    event_type = _safe_text(_get_field(header, "event_type", "")) or SUPPORTED_EVENT_TYPE
    if event_type and event_type != SUPPORTED_EVENT_TYPE:
        return None

    raw = {
        "event": {
            "sender": {
                "sender_id": _sender_id_dict(sender_id),
                "sender_type": _safe_text(_get_field(sender, "sender_type", "")),
                "tenant_key": _safe_text(_get_field(sender, "tenant_key", "")),
            },
            "message": {
                "message_id": _safe_text(_get_field(message, "message_id", "")),
                "root_id": _safe_text(_get_field(message, "root_id", "")),
                "parent_id": _safe_text(_get_field(message, "parent_id", "")),
                "create_time": _safe_text(_get_field(message, "create_time", "")),
                "update_time": _safe_text(_get_field(message, "update_time", "")),
                "chat_id": _safe_text(_get_field(message, "chat_id", "")),
                "thread_id": _safe_text(_get_field(message, "thread_id", "")),
                "chat_type": _safe_text(_get_field(message, "chat_type", "")),
                "message_type": _safe_text(_get_field(message, "message_type", "")),
                "content": _safe_text(_get_field(message, "content", "")),
                "mentions": _mentions_list(_get_field(message, "mentions", [])),
                "user_agent": _safe_text(_get_field(message, "user_agent", "")),
            },
        },
    }

    return MessageEvent(
        event_type=event_type,
        event_id=_safe_text(_get_field(header, "event_id", "")),
        create_time=_safe_text(_get_field(header, "create_time", "")) or raw["event"]["message"]["create_time"],
        message_id=raw["event"]["message"]["message_id"],
        root_id=raw["event"]["message"]["root_id"],
        parent_id=raw["event"]["message"]["parent_id"],
        update_time=raw["event"]["message"]["update_time"],
        chat_id=raw["event"]["message"]["chat_id"],
        thread_id=raw["event"]["message"]["thread_id"],
        chat_type=raw["event"]["message"]["chat_type"],
        message_type=raw["event"]["message"]["message_type"],
        content_raw=raw["event"]["message"]["content"],
        content_text=_parse_content_text(raw["event"]["message"]["content"]),
        mentions=raw["event"]["message"]["mentions"],
        user_agent=raw["event"]["message"]["user_agent"],
        sender_open_id=raw["event"]["sender"]["sender_id"]["open_id"],
        sender_union_id=raw["event"]["sender"]["sender_id"]["union_id"],
        sender_user_id=raw["event"]["sender"]["sender_id"]["user_id"],
        sender_type=raw["event"]["sender"]["sender_type"],
        tenant_key=raw["event"]["sender"]["tenant_key"],
        raw=raw,
    )


def should_accept_message_event(event: MessageEvent) -> bool:
    if event.sender_type != "user":
        return False
    return event.chat_type in {"p2p", "group"}


def dispatch_message_event(bus: MessageEventBus, event: MessageEvent) -> None:
    bus.publish(TOPIC_MESSAGE_RECEIVED, event)
    if event.chat_type == "p2p":
        bus.publish(TOPIC_MESSAGE_USER_P2P, event)
    elif event.chat_type == "group":
        bus.publish(TOPIC_MESSAGE_USER_GROUP, event)


def resolve_listener_credentials(env_file: str = "") -> tuple[str, str]:
    load_listener_env(env_file)
    app_id = (
        _safe_text(os.getenv("LISTENER_APP_ID"))
        or _safe_text(os.getenv("SOFREE_FEISHU_APP_ID"))
        or _safe_text(os.getenv("FEISHU_APP_ID"))
    )
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


class OpenAPIMessageListener:
    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        bus: MessageEventBus,
        compact: bool = True,
        print_events: bool = True,
        log_level: lark.LogLevel = lark.LogLevel.INFO,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.bus = bus
        self.compact = compact
        self.print_events = print_events
        self._event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_message_receive)
            .build()
        )
        ws_factory = client_factory or lark.ws.Client
        self._client = ws_factory(
            app_id,
            app_secret,
            event_handler=self._event_handler,
            log_level=log_level,
        )

    def start(self) -> None:
        self._client.start()

    def _handle_message_receive(self, data: Any) -> None:
        event = parse_message_event(data)
        if event is None:
            return
        if not should_accept_message_event(event):
            LOGGER.debug(
                "drop message event: message_id=%s sender_type=%s chat_type=%s",
                event.message_id,
                event.sender_type,
                event.chat_type,
            )
            return
        dispatch_message_event(self.bus, event)
        if self.print_events:
            payload = event.to_compact_dict() if self.compact else event.to_dict()
            print(json.dumps(payload, ensure_ascii=False), flush=True)
