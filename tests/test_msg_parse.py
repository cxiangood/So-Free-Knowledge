from __future__ import annotations

import json

from insight.msg.parse import event_row_to_message_event, plain_message_to_event
from insight.msg.types import PlainMessage


def test_event_row_to_message_event_preserves_sender_name() -> None:
    row = {
        "sender": {
            "sender_id": {"union_id": "on-1", "user_id": "u-1", "open_id": "ou-1"},
            "sender_type": "user",
            "tenant_key": "tenant",
            "name": "Alice",
        },
        "message": {
            "message_id": "om-1",
            "chat_id": "oc-1",
            "chat_type": "group",
            "message_type": "text",
            "content": json.dumps({"text": "hello"}, ensure_ascii=False),
            "mentions": [],
        },
    }

    event = event_row_to_message_event(row)

    assert event is not None
    assert event.sender_name == "Alice"
    assert event.get_simple_message() == "[Alice] hello"


def test_plain_message_to_event_uses_sender_as_sender_name() -> None:
    message = PlainMessage(
        message_id="om-2",
        chat_id="oc-1",
        send_time="1777000000000",
        sender="Bob",
        mentions=[],
        content="offline hello",
    )

    event = plain_message_to_event(message)

    assert event.sender_name == "Bob"
    assert event.get_simple_message() == "[Bob] offline hello"
    assert event.to_dict()["sender"]["name"] == "Bob"
