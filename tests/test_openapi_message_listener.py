from __future__ import annotations

import json
from dataclasses import dataclass

from local_pipeline.message_event_bus import (
    MessageEventBus,
    TOPIC_MESSAGE_RECEIVED,
    TOPIC_MESSAGE_USER_GROUP,
    TOPIC_MESSAGE_USER_P2P,
)
from local_pipeline.openapi_message_listener import (
    OpenAPIMessageListener,
    dispatch_message_event,
    parse_message_event,
    should_accept_message_event,
)


@dataclass
class _SenderId:
    open_id: str


@dataclass
class _Sender:
    sender_type: str
    sender_id: _SenderId


@dataclass
class _Message:
    message_id: str
    chat_id: str
    chat_type: str
    message_type: str
    content: str
    create_time: str = ""


@dataclass
class _Header:
    event_id: str
    event_type: str
    create_time: str


@dataclass
class _Event:
    message: _Message
    sender: _Sender


@dataclass
class _Data:
    header: _Header
    event: _Event


def _build_data(chat_type: str = "p2p", sender_type: str = "user") -> _Data:
    return _Data(
        header=_Header(event_id="evt-1", event_type="im.message.receive_v1", create_time="1773491924409"),
        event=_Event(
            message=_Message(
                message_id="om-1",
                chat_id="oc-1",
                chat_type=chat_type,
                message_type="text",
                content=json.dumps({"text": "hello"}, ensure_ascii=False),
            ),
            sender=_Sender(sender_type=sender_type, sender_id=_SenderId(open_id="ou-1")),
        ),
    )


def test_parse_message_event() -> None:
    event = parse_message_event(_build_data(chat_type="p2p"))
    assert event is not None
    assert event.event_type == "im.message.receive_v1"
    assert event.message_id == "om-1"
    assert event.chat_id == "oc-1"
    assert event.chat_type == "p2p"
    assert event.message_type == "text"
    assert event.content_text == "hello"
    assert event.sender_open_id == "ou-1"
    assert event.sender_type == "user"


def test_filter_non_user_message() -> None:
    event = parse_message_event(_build_data(chat_type="group", sender_type="bot"))
    assert event is not None
    assert should_accept_message_event(event) is False


def test_dispatch_p2p_and_group_topics() -> None:
    bus = MessageEventBus()
    received_topics: list[str] = []

    bus.subscribe(TOPIC_MESSAGE_RECEIVED, lambda _: received_topics.append(TOPIC_MESSAGE_RECEIVED))
    bus.subscribe(TOPIC_MESSAGE_USER_P2P, lambda _: received_topics.append(TOPIC_MESSAGE_USER_P2P))
    bus.subscribe(TOPIC_MESSAGE_USER_GROUP, lambda _: received_topics.append(TOPIC_MESSAGE_USER_GROUP))

    p2p_event = parse_message_event(_build_data(chat_type="p2p"))
    assert p2p_event is not None
    dispatch_message_event(bus, p2p_event)
    assert TOPIC_MESSAGE_RECEIVED in received_topics
    assert TOPIC_MESSAGE_USER_P2P in received_topics
    assert TOPIC_MESSAGE_USER_GROUP not in received_topics

    received_topics.clear()
    group_event = parse_message_event(_build_data(chat_type="group"))
    assert group_event is not None
    dispatch_message_event(bus, group_event)
    assert TOPIC_MESSAGE_RECEIVED in received_topics
    assert TOPIC_MESSAGE_USER_GROUP in received_topics
    assert TOPIC_MESSAGE_USER_P2P not in received_topics


def test_listener_callback_integration() -> None:
    events: list[str] = []
    bus = MessageEventBus()
    bus.subscribe(TOPIC_MESSAGE_USER_GROUP, lambda evt: events.append(evt.message_id))

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def start(self) -> None:
            return None

    listener = OpenAPIMessageListener(
        app_id="cli_xxx",
        app_secret="xxx",
        bus=bus,
        compact=True,
        print_events=False,
        client_factory=FakeClient,
    )
    listener._handle_message_receive(_build_data(chat_type="group"))
    assert events == ["om-1"]

