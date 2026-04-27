from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from .chat_message_store import ChatMessageStore
from .io_utils import append_jsonl, read_json, write_json
from .openapi_message_listener import MessageEvent
from .router import route_cards
from .semantic_lifter import lift_candidates
from .shared_types import PlainMessage, now_utc_iso
from .signal_detector import detect_candidates
from .stores import LocalStateStore
from .task_card_sender import TaskPushConfig, push_task_card


LOGGER = logging.getLogger(__name__)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _preview(value: Any, *, max_len: int = 500, max_items: int = 20) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_len else value[: max_len - 3] + "..."
    if isinstance(value, list):
        items = value[:max_items]
        out = [_preview(item, max_len=max_len, max_items=max_items) for item in items]
        if len(value) > max_items:
            out.append({"_truncated_items": len(value) - max_items})
        return out
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["_truncated_keys"] = len(value) - max_items
                break
            out[str(k)] = _preview(v, max_len=max_len, max_items=max_items)
        return out
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _preview(value.to_dict(), max_len=max_len, max_items=max_items)
        except Exception:
            return _safe_text(value)
    if hasattr(value, "__dict__"):
        return _preview(dict(value.__dict__), max_len=max_len, max_items=max_items)
    return _safe_text(value)


def _parse_content_text(raw_content: Any) -> str:
    content = _safe_text(raw_content)
    if not content:
        return ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return _safe_text(payload.get("text"))
    return content


def _mentions_to_names(mentions: Any) -> list[str]:
    if not isinstance(mentions, list):
        return []
    output: list[str] = []
    for item in mentions:
        if not isinstance(item, dict):
            continue
        name = _safe_text(item.get("name", ""))
        if name:
            output.append(name)
    return output


def _event_row_to_plain_message(row: dict[str, Any]) -> PlainMessage | None:
    event = row.get("event")
    if not isinstance(event, dict):
        return None
    sender = event.get("sender")
    message = event.get("message")
    if not isinstance(sender, dict) or not isinstance(message, dict):
        return None
    sender_id = sender.get("sender_id")
    if not isinstance(sender_id, dict):
        sender_id = {}

    message_id = _safe_text(message.get("message_id", ""))
    chat_id = _safe_text(message.get("chat_id", ""))
    if not message_id or not chat_id:
        return None
    content_raw = _safe_text(message.get("content", ""))
    return PlainMessage(
        message_id=message_id,
        chat_id=chat_id,
        send_time=_safe_text(message.get("create_time", "")),
        sender=_safe_text(sender_id.get("open_id", "")),
        mentions=_mentions_to_names(message.get("mentions", [])),
        content=_parse_content_text(content_raw),
        msg_type=_safe_text(message.get("message_type", "")) or "text",
    )


@dataclass(slots=True)
class RealtimeProcessResult:
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
        return asdict(self)


class ListenerRuntimeState:
    def __init__(self, path: str | Path, *, max_processed_ids: int = 2000) -> None:
        self.path = Path(path)
        self.max_processed_ids = max(100, int(max_processed_ids or 2000))
        self._lock = RLock()

    def contains(self, message_id: str) -> bool:
        normalized = _safe_text(message_id)
        if not normalized:
            return False
        with self._lock:
            data = self._load()
            return normalized in data["processed_ids"]

    def add(self, message_id: str) -> None:
        normalized = _safe_text(message_id)
        if not normalized:
            return
        with self._lock:
            data = self._load()
            ids = data["processed_ids"]
            if normalized in ids:
                return
            ids.append(normalized)
            if len(ids) > self.max_processed_ids:
                data["processed_ids"] = ids[-self.max_processed_ids :]
            write_json(self.path, data)

    def _load(self) -> dict[str, list[str]]:
        raw = read_json(self.path, default={"processed_ids": []})
        if not isinstance(raw, dict):
            return {"processed_ids": []}
        values = raw.get("processed_ids", [])
        if not isinstance(values, list):
            return {"processed_ids": []}
        ids = [_safe_text(item) for item in values if _safe_text(item)]
        return {"processed_ids": ids}


@dataclass(slots=True)
class RealtimeProcessorConfig:
    state_dir: str | Path = "outputs/local_pipeline/state"
    output_dir: str | Path = "outputs/local_pipeline"
    context_window_size: int = 20
    enable_llm: bool = False
    candidate_threshold: float = 0.45
    knowledge_threshold: float = 0.60
    task_threshold: float = 0.50
    task_push_enabled: bool = False
    task_push_chat_id: str = ""
    env_file: str = ""
    step_trace_enabled: bool = True


class RealtimeProcessor:
    def __init__(
        self,
        *,
        chat_store: ChatMessageStore,
        config: RealtimeProcessorConfig,
        state_store: LocalStateStore | None = None,
        runtime_state: ListenerRuntimeState | None = None,
    ) -> None:
        self.chat_store = chat_store
        self.config = config
        self.state_store = state_store or LocalStateStore(config.state_dir)
        self.runtime_state = runtime_state or ListenerRuntimeState(Path(config.state_dir) / "listener_runtime_state.json")
        self._events_path = Path(config.state_dir) / "realtime_events.jsonl"
        self._step_trace_enabled = bool(config.step_trace_enabled)

    def process_incoming_event(self, event: MessageEvent) -> RealtimeProcessResult:
        result = RealtimeProcessResult(message_id=event.message_id, chat_id=event.chat_id)
        self._emit_step(
            event,
            "start",
            status="ok",
            input_payload={
                "event": event.to_dict(),
                "config": {
                    "context_window_size": self.config.context_window_size,
                    "candidate_threshold": self.config.candidate_threshold,
                    "knowledge_threshold": self.config.knowledge_threshold,
                    "task_threshold": self.config.task_threshold,
                    "enable_llm": self.config.enable_llm,
                    "task_push_enabled": self.config.task_push_enabled,
                },
            },
            output_payload={"message_id": event.message_id, "chat_id": event.chat_id},
        )
        if self.runtime_state.contains(event.message_id):
            result.skipped = True
            self._emit_step(
                event,
                "deduplicate",
                status="skipped",
                detail="message_id already processed",
                input_payload={"message_id": event.message_id},
                output_payload={"skipped": True},
            )
            self._append_result(result)
            return result

        context = self.chat_store.get_chat_messages(event.chat_id)
        window = max(1, int(self.config.context_window_size or 20))
        context = context[-window:]
        self._emit_step(
            event,
            "context_loaded",
            status="ok",
            input_payload={"chat_id": event.chat_id, "window_size": window},
            output_payload={"context_size": len(context), "context_rows": context},
        )
        messages: list[PlainMessage] = []
        for row in context:
            converted = _event_row_to_plain_message(row)
            if converted is not None:
                messages.append(converted)
        self._emit_step(
            event,
            "context_mapped",
            status="ok",
            input_payload={"context_size": len(context)},
            output_payload={"plain_messages": [item.to_dict() for item in messages]},
        )

        detection = detect_candidates(messages, candidate_threshold=self.config.candidate_threshold)
        target_candidates = [item for item in detection.candidates if event.message_id in item.source_message_ids]
        result.candidate_count = len(target_candidates)
        self._emit_step(
            event,
            "signal_detected",
            status="ok",
            detail=f"candidates_total={len(detection.candidates)} current_message_candidates={len(target_candidates)}",
            input_payload={"plain_messages_count": len(messages)},
            output_payload={
                "detection_messages": [item.to_dict() for item in detection.messages],
                "all_candidates": [item.to_dict() for item in detection.candidates],
                "current_message_candidates": [item.to_dict() for item in target_candidates],
            },
        )
        if not target_candidates:
            self._emit_step(
                event,
                "short_circuit",
                status="ok",
                detail="no candidate for current message",
                input_payload={"message_id": event.message_id},
                output_payload={"reason": "no_candidate_for_current_message"},
            )
            self.runtime_state.add(event.message_id)
            self._emit_step(
                event,
                "runtime_state_updated",
                status="ok",
                input_payload={"message_id": event.message_id},
                output_payload={"processed": True},
            )
            self._append_result(result)
            self._emit_step(
                event,
                "event_recorded",
                status="ok",
                input_payload={"result": result.to_dict()},
                output_payload={"events_path": str(self._events_path)},
            )
            return result

        lift_result = lift_candidates(
            target_candidates,
            detection.messages,
            enable_llm=bool(self.config.enable_llm),
        )
        result.warnings.extend(lift_result.warnings)
        self._emit_step(
            event,
            "semantic_lifted",
            status="ok",
            detail=f"cards={len(lift_result.cards)} warnings={len(lift_result.warnings)}",
            input_payload={"candidates": [item.to_dict() for item in target_candidates], "enable_llm": self.config.enable_llm},
            output_payload={"cards": [item.to_dict() for item in lift_result.cards], "warnings": lift_result.warnings},
        )

        decisions = route_cards(
            lift_result.cards,
            knowledge_threshold=self.config.knowledge_threshold,
            task_threshold=self.config.task_threshold,
        )
        route_counts = Counter(item.target_pool for item in decisions)
        result.routed_counts = dict(route_counts)
        self._emit_step(
            event,
            "routed",
            status="ok",
            detail=f"route_counts={dict(route_counts)}",
            input_payload={"cards": [item.to_dict() for item in lift_result.cards]},
            output_payload={"decisions": [item.to_dict() for item in decisions], "route_counts": dict(route_counts)},
        )

        push_cfg = TaskPushConfig(
            enabled=bool(self.config.task_push_enabled),
            chat_id=str(self.config.task_push_chat_id or ""),
            env_file=str(self.config.env_file or ""),
        )
        pending_rows: list[dict[str, Any]] = []

        for decision in decisions:
            card = next((item for item in lift_result.cards if item.card_id == decision.card_id), None)
            if card is None:
                continue
            if decision.target_pool == "knowledge":
                decision.stored_id = self.state_store.add_knowledge(card)
                self._emit_step(
                    event,
                    "stored_knowledge",
                    status="ok",
                    detail=f"knowledge_id={decision.stored_id}",
                    input_payload={"decision": decision.to_dict(), "card": card.to_dict()},
                    output_payload={"knowledge_id": decision.stored_id},
                )
            elif decision.target_pool == "task":
                decision.stored_id = self.state_store.add_task(card)
                self._emit_step(
                    event,
                    "stored_task",
                    status="ok",
                    detail=f"task_id={decision.stored_id}",
                    input_payload={"decision": decision.to_dict(), "card": card.to_dict()},
                    output_payload={"task_id": decision.stored_id},
                )
                result.task_push_attempted += 1
                run_id = self._build_realtime_run_id(event)
                attempt = push_task_card(
                    config=push_cfg,
                    run_id=run_id,
                    task_id=decision.stored_id,
                    card=card,
                )
                if attempt.status == "sent":
                    result.task_push_sent += 1
                    self._emit_step(
                        event,
                        "task_pushed",
                        status="sent",
                        detail=f"task_id={decision.stored_id}",
                        input_payload={"run_id": run_id, "task_id": decision.stored_id, "chat_id": push_cfg.chat_id},
                        output_payload=attempt.to_dict(),
                    )
                elif attempt.status == "failed":
                    result.task_push_failed += 1
                    result.errors.append(_safe_text(attempt.error))
                    pending_rows.append(attempt.to_pending_dict())
                    self._emit_step(
                        event,
                        "task_pushed",
                        status="failed",
                        detail=f"task_id={decision.stored_id} error={_safe_text(attempt.error)}",
                        input_payload={"run_id": run_id, "task_id": decision.stored_id, "chat_id": push_cfg.chat_id},
                        output_payload=attempt.to_dict(),
                    )
                else:
                    self._emit_step(
                        event,
                        "task_pushed",
                        status=attempt.status,
                        detail=f"task_id={decision.stored_id} error={_safe_text(attempt.error)}",
                        input_payload={"run_id": run_id, "task_id": decision.stored_id, "chat_id": push_cfg.chat_id},
                        output_payload=attempt.to_dict(),
                    )
            else:
                decision.stored_id = self.state_store.add_observe(card)
                self._emit_step(
                    event,
                    "stored_observe",
                    status="ok",
                    detail=f"observe_id={decision.stored_id}",
                    input_payload={"decision": decision.to_dict(), "card": card.to_dict()},
                    output_payload={"observe_id": decision.stored_id},
                )

        if pending_rows:
            append_jsonl(Path(self.config.state_dir) / "pending_task_push.jsonl", pending_rows)
            self._emit_step(
                event,
                "pending_push_queued",
                status="ok",
                detail=f"count={len(pending_rows)}",
                input_payload={"failed_attempts": pending_rows},
                output_payload={"pending_count": len(pending_rows)},
            )

        self.runtime_state.add(event.message_id)
        self._emit_step(
            event,
            "runtime_state_updated",
            status="ok",
            input_payload={"message_id": event.message_id},
            output_payload={"processed": True},
        )
        self._append_result(result)
        self._emit_step(
            event,
            "event_recorded",
            status="ok",
            input_payload={"result": result.to_dict()},
            output_payload={"events_path": str(self._events_path)},
        )
        self._emit_step(
            event,
            "done",
            status="ok",
            detail=(
                f"candidate_count={result.candidate_count} routed={result.routed_counts} "
                f"task_push={result.task_push_attempted}/{result.task_push_sent}/{result.task_push_failed}"
            ),
            input_payload={"message_id": event.message_id},
            output_payload=result.to_dict(),
        )
        return result

    def _append_result(self, result: RealtimeProcessResult) -> None:
        append_jsonl(self._events_path, [result.to_dict()])

    def _emit_step(
        self,
        event: MessageEvent,
        step: str,
        *,
        status: str,
        detail: str = "",
        input_payload: Any = None,
        output_payload: Any = None,
    ) -> None:
        if not self._step_trace_enabled:
            return
        payload = {
            "type": "realtime_step",
            "message_id": event.message_id,
            "chat_id": event.chat_id,
            "step": step,
            "status": status,
            "detail": detail,
            "input": _preview(input_payload),
            "output": _preview(output_payload),
            "ts": now_utc_iso(),
        }
        line = json.dumps(payload, ensure_ascii=False)
        LOGGER.info(line)
        print(line, flush=True)

    @staticmethod
    def _build_realtime_run_id(event: MessageEvent) -> str:
        seed = f"{event.chat_id}:{event.message_id}:{event.create_time}"
        return "rt-" + hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]
