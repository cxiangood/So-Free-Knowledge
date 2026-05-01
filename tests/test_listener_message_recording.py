from __future__ import annotations

import json
from pathlib import Path

from local_pipeline.listener_service import ListenerService, ListenerServiceConfig
from local_pipeline.message_event_bus import TOPIC_MESSAGE_RECEIVED
from local_pipeline.openapi_message_listener import MessageEvent
from local_pipeline.io_utils import read_json


def test_listener_records_messages_on_received_topic(tmp_path: Path, monkeypatch) -> None:
    class FakeListener:
        def __init__(self, *, bus, **kwargs):
            self.bus = bus
            self.kwargs = kwargs

        def start(self) -> None:
            self.bus.publish(
                TOPIC_MESSAGE_RECEIVED,
                MessageEvent(
                    event_type="im.message.receive_v1",
                    event_id="evt-1",
                    create_time="1773491924409",
                    message_id="om-1",
                    root_id="",
                    parent_id="",
                    update_time="",
                    chat_id="oc-1",
                    thread_id="",
                    chat_type="group",
                    message_type="text",
                    content_text="这个问题今天要处理吗?",
                    content_raw='{"text":"这个问题今天要处理吗?"}',
                    sender_open_id="ou-1",
                    sender_union_id="on-1",
                    sender_user_id="u-1",
                    sender_type="user",
                    tenant_key="tk-1",
                ),
            )

    monkeypatch.setattr("local_pipeline.listener_service.resolve_listener_credentials", lambda _: ("cli", "sec"))
    monkeypatch.setattr("local_pipeline.listener_service.OpenAPIMessageListener", FakeListener)

    store_path = tmp_path / "chat_message_store.json"
    service = ListenerService(
        ListenerServiceConfig(
            env_file="",
            event_types="im.message.receive_v1",
            compact=True,
            print_events=False,
            output_dir=str(tmp_path / "out"),
            state_dir=str(tmp_path / "state"),
            chat_history_path=str(store_path),
            chat_history_limit=100,
        )
    )
    service.start()

    payload = json.loads(store_path.read_text(encoding="utf-8"))
    assert "oc-1" in payload
    assert len(payload["oc-1"]) == 1
    assert payload["oc-1"][0]["event"]["message"]["message_id"] == "om-1"

    realtime_events = (tmp_path / "state" / "realtime_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert realtime_events
    assert json.loads(realtime_events[-1])["message_id"] == "om-1"

    observe = read_json(tmp_path / "state" / "observe_store.json", default={"items": []})
    assert isinstance(observe, dict)
    assert observe.get("items")
