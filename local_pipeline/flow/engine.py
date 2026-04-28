from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..msg.types import MessageEvent
from ..comm.send import TaskPushAttempt, TaskPushConfig, queue_failed_pushes
from ..core.detect import detect_candidates
from ..core.kb import save_knowledge
from ..core.lift import lift_candidates
from ..core.obs import save_observe
from ..core.route import route_cards
from ..core.task import save_task
from ..msg.cache import ChatMessageStore
from ..msg.parse import event_row_to_plain_message
from ..shared.utils import now_utc_iso
from ..store.io import append_jsonl, read_json, write_json
from ..store.state import LocalStateStore
from .trace import trace_finish, trace_node, trace_start


@dataclass(slots=True)
class EngineConfig:
    output_dir: str | Path = "outputs/local_pipeline"
    state_dir: str | Path = "outputs/local_pipeline/state"
    chat_history_path: str | Path = "outputs/local_pipeline/state/chat_message_store.json"
    chat_history_limit: int = 100
    context_window_size: int = 20
    enable_llm: bool = False
    candidate_threshold: float = 0.45
    knowledge_threshold: float = 0.60
    task_threshold: float = 0.50
    task_push_enabled: bool = False
    task_push_chat_id: str = ""
    env_file: str = ""
    step_trace_enabled: bool = True


@dataclass(slots=True)
class EngineResult:
    message_id: str
    chat_id: str
    candidate_count: int = 0
    routed_counts: dict[str, int] = field(default_factory=dict)
    task_push_attempted: int = 0
    task_push_sent: int = 0
    task_push_failed: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    created_at: str = field(default_factory=now_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "candidate_count": self.candidate_count,
            "routed_counts": self.routed_counts,
            "task_push_attempted": self.task_push_attempted,
            "task_push_sent": self.task_push_sent,
            "task_push_failed": self.task_push_failed,
            "errors": self.errors,
            "warnings": self.warnings,
            "skipped": self.skipped,
            "created_at": self.created_at,
        }


class RuntimeState:
    def __init__(self, path: str | Path, max_processed_ids: int = 2000) -> None:
        self.path = Path(path)
        self.max_processed_ids = max(100, int(max_processed_ids))

    def contains(self, message_id: str) -> bool:
        data = read_json(self.path, {"processed_ids": []})
        if not isinstance(data, dict):
            return False
        ids = data.get("processed_ids", [])
        return message_id in ids if isinstance(ids, list) else False

    def add(self, message_id: str) -> None:
        data = read_json(self.path, {"processed_ids": []})
        if not isinstance(data, dict):
            data = {"processed_ids": []}
        ids = data.get("processed_ids", [])
        if not isinstance(ids, list):
            ids = []
        if message_id not in ids:
            ids.append(message_id)
        data["processed_ids"] = ids[-self.max_processed_ids :]
        write_json(self.path, data)


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.chat_store = ChatMessageStore(path=config.chat_history_path, max_messages_per_chat=config.chat_history_limit)
        self.state_store = LocalStateStore(config.state_dir)
        self.runtime_state = RuntimeState(Path(config.state_dir) / "listener_runtime_state.json")
        self.events_path = Path(config.state_dir) / "realtime_events.jsonl"

    def run(self, message: MessageEvent, context: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> EngineResult:
        del context
        del config
        result = EngineResult(message_id=message.message_id, chat_id=message.chat_id)
        trace_status = "ok"
        if self.config.step_trace_enabled:
            trace_start(message_id=message.message_id, chat_id=message.chat_id, content=message.content_text)
        try:
            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="message_cache")
            self.chat_store.append(message)

            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="deduplicate")
            if self.runtime_state.contains(message.message_id):
                result.skipped = True
                trace_status = "skipped"
                self._append_result(result)
                return result

            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="context_extract")
            context_rows = self.chat_store.get_chat_messages(message.chat_id)[-max(1, int(self.config.context_window_size)) :]
            plain_messages = [item for item in (event_row_to_plain_message(row) for row in context_rows) if item is not None]

            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="signal_detect")
            detection = detect_candidates(plain_messages, candidate_threshold=self.config.candidate_threshold)
            current_candidates = [item for item in detection.candidates if message.message_id in item.source_message_ids]
            result.candidate_count = len(current_candidates)
            if not current_candidates:
                self.runtime_state.add(message.message_id)
                self._append_result(result)
                return result

            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="semantic_lift")
            lift_result = lift_candidates(current_candidates, detection.messages, enable_llm=self.config.enable_llm)
            result.warnings.extend(lift_result.warnings)

            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="route")
            decisions = route_cards(
                lift_result.cards,
                knowledge_threshold=self.config.knowledge_threshold,
                task_threshold=self.config.task_threshold,
            )
            for decision in decisions:
                result.routed_counts[decision.target_pool] = result.routed_counts.get(decision.target_pool, 0) + 1

            push_cfg = TaskPushConfig(
                enabled=self.config.task_push_enabled,
                chat_id=self.config.task_push_chat_id,
                env_file=self.config.env_file,
            )
            failed_attempts: list[TaskPushAttempt] = []
            for decision in decisions:
                card = next((item for item in lift_result.cards if item.card_id == decision.card_id), None)
                if card is None:
                    continue
                if decision.target_pool == "knowledge":
                    if self.config.step_trace_enabled:
                        trace_node(message_id=message.message_id, node_name="knowledge_store")
                    decision.stored_id = save_knowledge(self.state_store, card)
                elif decision.target_pool == "task":
                    if self.config.step_trace_enabled:
                        trace_node(message_id=message.message_id, node_name="task_store")
                    task_result = save_task(store=self.state_store, card=card, run_id=self._build_run_id(message), push_config=push_cfg)
                    decision.stored_id = task_result.task_id
                    result.task_push_attempted += 1
                    if task_result.push_attempt is not None:
                        if self.config.step_trace_enabled:
                            trace_node(message_id=message.message_id, node_name="task_push")
                        if task_result.push_attempt.status == "sent":
                            result.task_push_sent += 1
                        elif task_result.push_attempt.status == "failed":
                            result.task_push_failed += 1
                            trace_status = "failed"
                            result.errors.append(task_result.push_attempt.error)
                            failed_attempts.append(task_result.push_attempt)
                else:
                    if self.config.step_trace_enabled:
                        trace_node(message_id=message.message_id, node_name="observe_store")
                    decision.stored_id = save_observe(self.state_store, card)
            if failed_attempts:
                queue_failed_pushes(self.config.state_dir, failed_attempts)

            self.runtime_state.add(message.message_id)
            self._append_result(result)
            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="done")
            return result
        finally:
            if self.config.step_trace_enabled:
                trace_finish(message_id=message.message_id, suffix_status=trace_status)

    def _append_result(self, result: EngineResult) -> None:
        append_jsonl(self.events_path, [result.to_dict()])

    @staticmethod
    def _build_run_id(message: MessageEvent) -> str:
        return f"rt-{message.message_id}"


__all__ = ["Engine", "EngineConfig", "EngineResult"]
