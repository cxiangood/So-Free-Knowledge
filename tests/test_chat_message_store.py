from __future__ import annotations

import json
from pathlib import Path

from local_pipeline.chat_message_store import ChatMessageStore
from local_pipeline.io_utils import read_json
from local_pipeline.openapi_message_listener import MessageEvent


def _build_event(chat_id: str, idx: int) -> MessageEvent:
    return MessageEvent(
        event_type="im.message.receive_v1",
        message_id=f"om-{idx}",
        event_id=f"evt-{idx}",
        create_time=f"1773491924{idx}",
        root_id=f"om-root-{idx}",
        parent_id=f"om-parent-{idx}",
        update_time=f"1773492924{idx}",
        chat_id=chat_id,
        thread_id=f"omt-{idx}",
        chat_type="group",
        message_type="text",
        content_text=f"hello-{idx}",
        content_raw=json.dumps({"text": f"hello-{idx}"}, ensure_ascii=False),
        sender_open_id=f"ou-{idx}",
        sender_union_id=f"on-{idx}",
        sender_user_id=f"u-{idx}",
        sender_type="user",
        tenant_key="tk-1",
    )


def test_store_keeps_latest_messages_with_limit(tmp_path: Path) -> None:
    path = tmp_path / "chat_message_store.json"
    store = ChatMessageStore(path, max_messages_per_chat=3)

    for idx in range(5):
        store.append(_build_event("oc-a", idx))

    payload = read_json(path, default={})
    rows = payload["oc-a"]
    assert len(rows) == 3
    assert [item["event"]["message"]["message_id"] for item in rows] == ["om-2", "om-3", "om-4"]


def test_store_isolated_by_chat_id(tmp_path: Path) -> None:
    path = tmp_path / "chat_message_store.json"
    store = ChatMessageStore(path, max_messages_per_chat=2)
    store.append(_build_event("oc-a", 1))
    store.append(_build_event("oc-b", 2))
    store.append(_build_event("oc-a", 3))

    payload = read_json(path, default={})
    assert [item["event"]["message"]["message_id"] for item in payload["oc-a"]] == ["om-1", "om-3"]
    assert [item["event"]["message"]["message_id"] for item in payload["oc-b"]] == ["om-2"]


def test_store_loads_existing_payload_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "chat_message_store.json"
    seed = {"oc-a": [_build_event("oc-a", 1).to_dict()]}
    path.write_text(json.dumps(seed, ensure_ascii=False), encoding="utf-8")

    store = ChatMessageStore(path, max_messages_per_chat=3)
    store.append(_build_event("oc-a", 2))

    payload = read_json(path, default={})
    assert [item["event"]["message"]["message_id"] for item in payload["oc-a"]] == ["om-1", "om-2"]
