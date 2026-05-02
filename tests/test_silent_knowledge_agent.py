from __future__ import annotations

import json
import shutil
from pathlib import Path

from local_pipeline.agent import SilentKnowledgeAgent, SilentKnowledgeAgentConfig
from local_pipeline.agent.planner import SilentKnowledgePlanner
from local_pipeline.flow.engine import EngineConfig
from local_pipeline.flow.engine import EngineResult
from local_pipeline.flow.offline import OfflineConfig, run as run_offline
from local_pipeline.msg.types import MessageEvent


TEST_ROOT = Path("outputs/test_silent_knowledge_agent")


def _case_dir(name: str) -> Path:
    path = TEST_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _event(message_id: str, text: str, *, chat_id: str = "oc-agent") -> MessageEvent:
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


def test_silent_agent_handles_message_without_user_command(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: ["llm_api_key"])
    tmp_path = _case_dir("single_message")
    agent = SilentKnowledgeAgent(
        SilentKnowledgeAgentConfig(
            engine=EngineConfig(
                output_dir=tmp_path / "out",
                state_dir=tmp_path / "state",
                chat_history_path=tmp_path / "state" / "chat_message_store.json",
                step_trace_enabled=False,
                rag_enabled=False,
                task_threshold=0.40,
            )
        )
    )

    report = agent.handle_message(_event("om-agent-1", "这个问题需要今天安排修复吗？"))

    assert report.agent_name == "sofree-silent-knowledge-agent"
    assert report.trigger == "silent_message"
    assert report.message_id == "om-agent-1"
    assert report.tool_calls
    assert report.tool_calls[0].tool_name == "observe_message"
    assert report.decisions
    assert report.engine_result.candidate_count >= 1
    assert (tmp_path / "state" / "realtime_events.jsonl").exists()


def test_offline_run_uses_agent_report_shape(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: ["llm_api_key"])
    tmp_path = _case_dir("offline")
    messages = tmp_path / "messages.jsonl"
    messages.write_text(
        json.dumps(
            {
                "message_id": "m-agent-1",
                "chat_id": "oc-agent",
                "msg_type": "text",
                "create_time": "1777000000000",
                "sender": {"id": "u-1", "name": "Alice"},
                "content": "这个知识点后续可以沉淀一下吗？",
                "raw_content": json.dumps({"text": "这个知识点后续可以沉淀一下吗？"}, ensure_ascii=False),
                "mentions": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_offline(
        OfflineConfig(
            messages_file=messages,
            output_dir=tmp_path / "out",
            state_dir=tmp_path / "state",
            chat_history_path=tmp_path / "state" / "chat_message_store.json",
            step_trace_enabled=False,
            rag_enabled=False,
            task_threshold=0.40,
        )
    )

    assert result["ok"] is True
    assert result["agent_name"] == "sofree-silent-knowledge-agent"
    assert result["trigger"] == "manual_replay"
    assert result["mode"] == "offline"
    assert result["message_count"] == 1
    assert "route_counts" in result


def test_agent_exposes_tool_definitions() -> None:
    agent = SilentKnowledgeAgent(
        SilentKnowledgeAgentConfig(
            engine=EngineConfig(step_trace_enabled=False, rag_enabled=False)
        )
    )

    tools = agent.tool_definitions()

    names = {item["name"] for item in tools}
    assert {"observe_message", "retrieve_knowledge", "answer_from_memory", "push_text"}.issubset(names)
    observe = next(item for item in tools if item["name"] == "observe_message")
    assert observe["silent_safe"] is True
    assert any(param["name"] == "message" for param in observe["parameters"])


def test_planner_decides_wait_when_signal_is_weak() -> None:
    planner = SilentKnowledgePlanner(agent_name="test-agent")
    result = EngineResult(message_id="om-noise", chat_id="oc-agent", candidate_count=0)

    decisions = planner.decide_after_observation(result)

    assert decisions[0].action == "wait"
    assert decisions[0].tool_name == "none"
    assert "keep observing" in decisions[0].reason


def test_planner_decides_push_and_retry_from_engine_result() -> None:
    planner = SilentKnowledgePlanner(agent_name="test-agent")
    result = EngineResult(
        message_id="om-task",
        chat_id="oc-agent",
        candidate_count=1,
        routed_counts={"task": 1},
        task_push_sent=1,
        task_push_failed=1,
        errors=["permission denied"],
    )

    decisions = planner.decide_after_observation(result)

    actions = [item.action for item in decisions]
    assert "store" in actions
    assert "push" in actions
    assert "retry" in actions
