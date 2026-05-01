from __future__ import annotations

import json
from pathlib import Path

import local_pipeline.pipeline as pipeline_module
from local_pipeline.pipeline import PipelineConfig, run_pipeline
from local_pipeline.stores import LocalStateStore
from local_pipeline.task_card_sender import TaskPushAttempt, _resolve_sender_credentials


def _write_messages(path: Path, contents: list[tuple[str, str]]) -> None:
    rows = []
    for idx, (msg_type, text) in enumerate(contents):
        rows.append(
            {
                "message_id": f"m-{idx}",
                "chat_id": "oc_test",
                "msg_type": msg_type,
                "create_time": str(1777000000000 + idx * 1000),
                "sender": {"id": f"u-{idx}", "name": f"user-{idx}"},
                "content": text,
                "raw_content": json.dumps({"text": text}, ensure_ascii=False),
                "mentions": [],
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_pipeline_run_generates_core_artifacts(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    _write_messages(
        messages,
        [
            ("system", "ignore"),
            ("text", "大家这个功能有点问题，建议尽快优化。"),
            ("post", "今天能不能确认下上线节奏？"),
            ("text", "This message was recalled"),
        ],
    )
    config = PipelineConfig(output_dir=tmp_path / "out", state_dir=tmp_path / "state")
    result = run_pipeline(messages_file=messages, run_id="run-a", enable_llm=False, config=config)

    run_dir = Path(result["run_dir"])
    assert result["ok"] is True
    assert (run_dir / "run_metrics.json").exists()
    assert (run_dir / "route_decisions.jsonl").exists()
    assert (run_dir / "push_events.jsonl").exists()
    assert (run_dir / "run_report.md").exists()


def test_enable_llm_missing_config_falls_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL_ID", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    messages = tmp_path / "messages.jsonl"
    _write_messages(messages, [("text", "大家这个问题需要尽快处理吗？")])
    config = PipelineConfig(output_dir=tmp_path / "out", state_dir=tmp_path / "state")
    result = run_pipeline(messages_file=messages, run_id="run-b", enable_llm=True, config=config)

    warnings = result.get("warnings", [])
    assert warnings
    assert "missing config" in warnings[0]


def test_observe_hit_count_can_escalate(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    _write_messages(messages, [("text", "线上问题出现崩溃！")])

    config = PipelineConfig(
        output_dir=tmp_path / "out",
        state_dir=tmp_path / "state",
        observe_escalation_threshold=2,
        candidate_threshold=0.42,
    )
    run_pipeline(messages_file=messages, run_id="run-c1", enable_llm=False, config=config)
    run_pipeline(messages_file=messages, run_id="run-c2", enable_llm=False, config=config)

    observe = json.loads((tmp_path / "state" / "observe_store.json").read_text(encoding="utf-8"))
    items = observe.get("items", [])
    assert items
    assert any(str(item.get("escalation_state", "")).startswith("escalated_") for item in items)


def test_feedback_updates_apply_to_tasks(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    _write_messages(messages, [("text", "@张三 这个需求今天可以安排实现吗？")])
    config = PipelineConfig(output_dir=tmp_path / "out", state_dir=tmp_path / "state")
    run_pipeline(messages_file=messages, run_id="run-d1", enable_llm=False, config=config)

    store = LocalStateStore(tmp_path / "state")
    snapshot = store.snapshot()
    tasks = snapshot["tasks"]
    assert tasks
    task_id = str(tasks[0]["task_id"])

    updates_file = tmp_path / "task_updates.jsonl"
    updates_file.write_text(
        json.dumps({"task_id": task_id, "status": "done", "comment": "完成"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = store.apply_task_updates(updates_file).to_dict()
    assert summary["updated_count"] >= 1
    assert summary["done_count"] >= 1


def test_task_route_triggers_push_and_updates_push_events(tmp_path: Path, monkeypatch) -> None:
    messages = tmp_path / "messages.jsonl"
    _write_messages(messages, [("text", "@张三 今天可以安排修复吗？")])

    def _fake_push_task_card(*, config, run_id, task_id, card):
        return TaskPushAttempt(
            task_id=task_id,
            run_id=run_id,
            chat_id=config.chat_id,
            card_payload={"mock": True},
            status="sent",
            message_id="om_sent_1",
            error="",
        )

    monkeypatch.setattr(pipeline_module, "push_task_card", _fake_push_task_card)
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        state_dir=tmp_path / "state",
        task_push_enabled=True,
        task_push_chat_id="oc_push_target",
    )
    result = run_pipeline(messages_file=messages, run_id="run-push-ok", enable_llm=False, config=config)
    assert result["task_push_attempted"] >= 1
    assert result["task_push_sent"] >= 1
    assert result["task_push_failed"] == 0

    push_events = (tmp_path / "out" / "run-push-ok" / "push_events.jsonl").read_text(encoding="utf-8").splitlines()
    parsed_events = [json.loads(line) for line in push_events if line.strip()]
    task_events = [event for event in parsed_events if event.get("target") == "task"]
    assert task_events
    assert task_events[0]["delivery_status"] == "sent"
    assert task_events[0]["message_id"] == "om_sent_1"


def test_task_push_failure_goes_to_pending_queue(tmp_path: Path, monkeypatch) -> None:
    messages = tmp_path / "messages.jsonl"
    _write_messages(messages, [("text", "@张三 今天可以安排修复吗？")])

    def _fake_push_task_card(*, config, run_id, task_id, card):
        return TaskPushAttempt(
            task_id=task_id,
            run_id=run_id,
            chat_id=config.chat_id,
            card_payload={"mock": True},
            status="failed",
            message_id="",
            error="permission denied",
        )

    monkeypatch.setattr(pipeline_module, "push_task_card", _fake_push_task_card)
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        state_dir=tmp_path / "state",
        task_push_enabled=True,
        task_push_chat_id="oc_push_target",
    )
    result = run_pipeline(messages_file=messages, run_id="run-push-fail", enable_llm=False, config=config)
    assert result["task_push_attempted"] >= 1
    assert result["task_push_failed"] >= 1
    assert result["task_push_failures"]

    pending_file = tmp_path / "state" / "pending_task_push.jsonl"
    assert pending_file.exists()
    pending = [json.loads(line) for line in pending_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert pending
    assert pending[0]["error"] == "permission denied"
    assert pending[0]["retry_count"] == 0


def test_card_sender_credential_priority(monkeypatch) -> None:
    monkeypatch.setenv("CARD_SENDER_APP_ID", "cli_card")
    monkeypatch.setenv("CARD_SENDER_APP_SECRET", "secret_card")
    monkeypatch.setenv("FEISHU_APP_ID", "cli_feishu")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_feishu")
    app_id, app_secret = _resolve_sender_credentials()
    assert app_id == "cli_card"
    assert app_secret == "secret_card"
