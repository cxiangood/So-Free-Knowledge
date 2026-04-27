from __future__ import annotations

import json
from pathlib import Path

from local_pipeline.chat_message_store import ChatMessageStore
from local_pipeline.openapi_message_listener import MessageEvent
from local_pipeline.realtime_processor import RealtimeProcessor, RealtimeProcessorConfig
from local_pipeline.task_card_sender import TaskPushAttempt


def _build_event(chat_id: str, message_id: str, content_text: str) -> MessageEvent:
    return MessageEvent(
        event_type="im.message.receive_v1",
        event_id=f"evt-{message_id}",
        create_time="1773491924409",
        message_id=message_id,
        root_id="",
        parent_id="",
        update_time="",
        chat_id=chat_id,
        thread_id="",
        chat_type="group",
        message_type="text",
        content_text=content_text,
        content_raw=json.dumps({"text": content_text}, ensure_ascii=False),
        sender_open_id="ou-x",
        sender_union_id="on-x",
        sender_user_id="u-x",
        sender_type="user",
        tenant_key="tk-1",
    )


def test_realtime_processor_only_processes_current_message_candidate(tmp_path: Path) -> None:
    chat_store = ChatMessageStore(tmp_path / "chat_message_store.json", max_messages_per_chat=100)
    first = _build_event("oc-a", "om-1", "这个问题需要处理吗?")
    second = _build_event("oc-a", "om-2", "这个任务今天安排吗?")
    chat_store.append(first)
    chat_store.append(second)

    processor = RealtimeProcessor(
        chat_store=chat_store,
        config=RealtimeProcessorConfig(
            state_dir=tmp_path / "state",
            output_dir=tmp_path / "out",
            context_window_size=20,
            candidate_threshold=0.45,
            task_threshold=0.40,
        ),
    )
    result = processor.process_incoming_event(second)
    assert result.skipped is False
    assert result.candidate_count == 1


def test_realtime_processor_deduplicates_same_message_id(tmp_path: Path) -> None:
    chat_store = ChatMessageStore(tmp_path / "chat_message_store.json", max_messages_per_chat=100)
    event = _build_event("oc-a", "om-1", "这个问题需要处理吗?")
    chat_store.append(event)
    processor = RealtimeProcessor(
        chat_store=chat_store,
        config=RealtimeProcessorConfig(
            state_dir=tmp_path / "state",
            output_dir=tmp_path / "out",
            context_window_size=20,
            candidate_threshold=0.45,
        ),
    )

    first = processor.process_incoming_event(event)
    second = processor.process_incoming_event(event)
    assert first.skipped is False
    assert second.skipped is True


def test_realtime_processor_task_route_triggers_push(tmp_path: Path, monkeypatch) -> None:
    pushed: list[str] = []

    def _fake_push_task_card(*, config, run_id, task_id, card):
        pushed.append(task_id)
        return TaskPushAttempt(
            task_id=task_id,
            run_id=run_id,
            chat_id=config.chat_id,
            card_payload={},
            status="sent",
            message_id="om-sent",
        )

    monkeypatch.setattr("local_pipeline.realtime_processor.push_task_card", _fake_push_task_card)

    chat_store = ChatMessageStore(tmp_path / "chat_message_store.json", max_messages_per_chat=100)
    event = _build_event("oc-a", "om-1", "这个任务今天安排吗?")
    chat_store.append(event)

    processor = RealtimeProcessor(
        chat_store=chat_store,
        config=RealtimeProcessorConfig(
            state_dir=tmp_path / "state",
            output_dir=tmp_path / "out",
            context_window_size=20,
            candidate_threshold=0.45,
            task_threshold=0.40,
            task_push_enabled=True,
            task_push_chat_id="oc-target",
        ),
    )
    result = processor.process_incoming_event(event)
    assert result.task_push_attempted >= 1
    assert result.task_push_sent >= 1
    assert pushed


def test_realtime_processor_enable_llm_without_config_warns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL_ID", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    chat_store = ChatMessageStore(tmp_path / "chat_message_store.json", max_messages_per_chat=100)
    event = _build_event("oc-a", "om-1", "这个问题需要处理吗?")
    chat_store.append(event)

    processor = RealtimeProcessor(
        chat_store=chat_store,
        config=RealtimeProcessorConfig(
            state_dir=tmp_path / "state",
            output_dir=tmp_path / "out",
            context_window_size=20,
            candidate_threshold=0.45,
            enable_llm=True,
        ),
    )
    result = processor.process_incoming_event(event)
    assert any("LLM disabled automatically due to missing config" in item for item in result.warnings)

