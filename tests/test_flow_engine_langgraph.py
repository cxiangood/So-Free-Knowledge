from __future__ import annotations

import json
import shutil
from pathlib import Path

from insight.comm.send import TaskPushAttempt
from insight.flow.engine import Engine, EngineConfig
from insight.msg.types import MessageEvent


TEST_ROOT = Path("outputs/test_flow_engine_langgraph")


def _event(message_id: str, text: str, *, chat_id: str = "oc-test") -> MessageEvent:
    return MessageEvent(
        event_type="im.message.receive_v1",
        event_id=f"evt-{message_id}",
        create_time="1777000000000",
        message_id=message_id,
        chat_id=chat_id,
        chat_type="group",
        message_type="text",
        content_text=text,
        content_raw=json.dumps({"text": text}, ensure_ascii=False),
        root_id="",
        parent_id="",
        update_time="",
        thread_id="",
        sender_open_id="ou-test",
        sender_union_id="on-test",
        sender_user_id="u-test",
        sender_type="user",
        tenant_key="tenant",
        sender_name="Alice",
    )


def _engine(tmp_path: Path, **overrides) -> Engine:
    config = EngineConfig(
        output_dir=tmp_path / "out",
        state_dir=tmp_path / "state",
        chat_history_path=tmp_path / "state" / "chat_message_store.json",
        step_trace_enabled=False,
        rag_enabled=False,
        **overrides,
    )
    return Engine(config)


def _case_dir(name: str) -> Path:
    path = TEST_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_engine_graph_deduplicates_message(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: ["llm_api_key"])
    tmp_path = _case_dir("deduplicates")
    engine = _engine(tmp_path)
    event = _event("om-1", "需要安排修复这个问题吗？")

    first = engine.run(event)
    second = engine.run(event)

    assert first.skipped is False
    assert second.skipped is True


def test_engine_graph_no_candidate_only_records_result(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: ["llm_api_key"])
    tmp_path = _case_dir("no_candidate")
    engine = _engine(tmp_path)

    result = engine.run(_event("om-2", "收到"))

    assert result.candidate_count == 0
    assert result.routed_counts == {}
    assert (tmp_path / "state" / "realtime_events.jsonl").exists()


def test_engine_graph_task_route_stores_task(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: ["llm_api_key"])
    tmp_path = _case_dir("task_route")
    engine = _engine(tmp_path)

    result = engine.run(_event("om-3", "需要安排修复这个问题吗？"))

    assert result.routed_counts.get("task", 0) >= 1
    assert result.task_push_attempted >= 1
    tasks = json.loads((tmp_path / "state" / "task_store.json").read_text(encoding="utf-8"))
    assert tasks["items"]


def test_engine_graph_multi_route_stores_each_target(monkeypatch) -> None:
    from insight.core.detect import DetectionResult
    from insight.core.lift import LiftResult
    from insight.shared.models import LiftedCard, RouteDecision

    card = LiftedCard(
        card_id="card-multi",
        candidate_id="cand-multi",
        title="修复导出失败并沉淀规则",
        summary="导出失败需要修复，同时权限规则可沉淀。",
        problem="导出失败影响用户使用。",
        suggestion="今天完成修复并更新文档。",
        target_audience="团队成员",
        evidence=["导出失败和权限改动有关"],
        tags=["actionable-signal", "novel-content"],
        confidence=0.86,
        suggested_target="task",
        source_message_ids=["current"],
    )
    monkeypatch.setattr("local_pipeline.flow.engine.detect_candidates", lambda messages: DetectionResult(messages=messages, value_score=90.0))
    monkeypatch.setattr("local_pipeline.flow.engine.lift_candidates", lambda messages: LiftResult(cards=[card], warnings=[]))
    monkeypatch.setattr(
        "local_pipeline.flow.engine.route_cards",
        lambda cards: [
            RouteDecision(card_id=card.card_id, target_pool="knowledge", reason_codes=["knowledge-semantic-signal"], threshold_snapshot={}),
            RouteDecision(card_id=card.card_id, target_pool="task", reason_codes=["task-semantic-signal"], threshold_snapshot={}),
        ],
    )
    tmp_path = _case_dir("multi_route")
    engine = _engine(tmp_path)

    result = engine.run(_event("om-multi", "导出失败需要修复，同时沉淀规则"))

    assert result.routed_counts.get("knowledge", 0) == 1
    assert result.routed_counts.get("task", 0) == 1
    knowledge = json.loads((tmp_path / "state" / "knowledge_store.json").read_text(encoding="utf-8"))
    tasks = json.loads((tmp_path / "state" / "task_store.json").read_text(encoding="utf-8"))
    assert knowledge["items"]
    assert tasks["items"]


def test_engine_graph_task_push_failure_queues_retry(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: ["llm_api_key"])
    tmp_path = _case_dir("push_failure")

    def _fake_push_task_card(*, config, run_id, task_id, card):
        del card
        return TaskPushAttempt(
            task_id=task_id,
            run_id=run_id,
            chat_id=config.chat_id,
            card_payload={"mock": True},
            status="failed",
            message_id="",
            error="permission denied",
        )

    monkeypatch.setattr("local_pipeline.core.task.push_task_card", _fake_push_task_card)
    engine = _engine(tmp_path, task_push_enabled=True, task_push_chat_id="oc-target")

    result = engine.run(_event("om-4", "需要安排修复这个问题吗？"))

    assert result.task_push_failed == 1
    pending = tmp_path / "state" / "pending_task_push.jsonl"
    assert pending.exists()
    assert "permission denied" in pending.read_text(encoding="utf-8")
