from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..msg.types import MessageEvent
from ..comm.send import TaskPushAttempt, TaskPushConfig, push_text_message, queue_failed_pushes
from ..core.denoise import denoise_messages
from ..core.detect import detect_candidates
from ..core.observe_ferment import (
    apply_logic1_on_observe_add,
    apply_logic2_on_knowledge,
    apply_logic3_on_task,
    pop_ready_items,
)
from ..core.observe_qa import is_question_with_llm, try_answer_with_rag
from ..core.kb import save_knowledge
from ..core.lift import lift_candidates
from ..core.obs import save_observe
from ..core.route import route_cards
from ..core.task import enhance_task_card_with_rag, save_task
from ..msg.cache import ChatMessageStore
from ..msg.parse import event_row_to_message_event
from ..rag.index import VectorKnowledgeStore
from ..rag.retriever import retrieve
from ..shared.models import LiftedCard, ObservePopItem, ObserveReplyEvent
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
    candidate_threshold: float = 0.45
    knowledge_threshold: float = 0.60
    task_threshold: float = 0.50
    task_push_enabled: bool = False
    task_push_chat_id: str = ""
    env_file: str = ""
    step_trace_enabled: bool = True
    rag_enabled: bool = True
    rag_top_k: int = 5
    rag_min_score: float = 0.35
    rag_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    observe_auto_reply_enabled: bool = True
    observe_ferment_threshold: float = 4.0
    observe_logic1_base: float = 1.0
    observe_logic2_base: float = 1.5
    observe_logic3_base: float = 2.0
    observe_force_non_observe_on_pop: bool = True


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
    rag_retrieval_count: int = 0
    observe_question_count: int = 0
    observe_answered_count: int = 0
    observe_fallback_count: int = 0
    observe_pop_count: int = 0
    observe_reroute_task_count: int = 0
    observe_reroute_knowledge_count: int = 0
    denoise_filtered_count: int = 0
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
            "rag_retrieval_count": self.rag_retrieval_count,
            "observe_question_count": self.observe_question_count,
            "observe_answered_count": self.observe_answered_count,
            "observe_fallback_count": self.observe_fallback_count,
            "observe_pop_count": self.observe_pop_count,
            "observe_reroute_task_count": self.observe_reroute_task_count,
            "observe_reroute_knowledge_count": self.observe_reroute_knowledge_count,
            "denoise_filtered_count": self.denoise_filtered_count,
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
        self.observe_reply_events_path = Path(config.state_dir) / "observe_reply_events.jsonl"
        self.observe_ferment_events_path = Path(config.state_dir) / "observe_ferment_events.jsonl"
        self.observe_pop_events_path = Path(config.state_dir) / "observe_pop_events.jsonl"
        self.vector_store: VectorKnowledgeStore | None = None
        if self.config.rag_enabled:
            try:
                self.vector_store = VectorKnowledgeStore(Path(config.state_dir) / "vector_kb", embed_model=self.config.rag_embed_model)
            except Exception:
                self.vector_store = None

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
            message_events = [item for item in (event_row_to_message_event(row) for row in context_rows) if item is not None]
            
            # message_events, dropped = denoise_messages(message_events)
            # result.denoise_filtered_count = int(dropped)
            # if self.config.step_trace_enabled:
            #     trace_node(message_id=message.message_id, node_name="message_denoise")
            
            simple_messages = [message.get_simple_message() for message in message_events]
            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="signal_detect")
            detection = detect_candidates(simple_messages, candidate_threshold=self.config.candidate_threshold)
            current_candidates = list(detection.candidates)
            result.candidate_count = len(current_candidates)
            if not current_candidates:
                self.runtime_state.add(message.message_id)
                self._append_result(result)
                return result

            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="semantic_lift")
            lift_result = lift_candidates(current_candidates, message_events)
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
                    decision.stored_id = save_knowledge(self.state_store, card, vector_store=self.vector_store)
                    if self.config.step_trace_enabled:
                        trace_node(message_id=message.message_id, node_name="observe_logic2_check")
                    ferment_results = apply_logic2_on_knowledge(
                        card=card,
                        state_store=self.state_store,
                        threshold=self.config.observe_ferment_threshold,
                        base_score=self.config.observe_logic2_base,
                    )
                    for item in ferment_results:
                        self._append_observe_ferment_event(message.message_id, card.card_id, item.to_dict())
                    self._process_observe_pop(message=message, result=result)
                elif decision.target_pool == "task":
                    task_card = card
                    if self.config.rag_enabled:
                        if self.config.step_trace_enabled:
                            trace_node(message_id=message.message_id, node_name="rag_retrieve_task")
                        query = f"{card.summary} {card.problem} {card.suggestion}".strip()
                        try:
                            hits = retrieve(
                                self.vector_store,
                                query=query,
                                top_k=self.config.rag_top_k,
                                min_score=self.config.rag_min_score,
                            )
                        except Exception as exc:
                            hits = []
                            result.warnings.append(f"task rag retrieval failed: {exc}")
                        if hits:
                            result.rag_retrieval_count += 1
                            task_card = enhance_task_card_with_rag(card, hits)
                    if self.config.step_trace_enabled:
                        trace_node(message_id=message.message_id, node_name="task_store")
                    task_result = save_task(store=self.state_store, card=task_card, run_id=self._build_run_id(message), push_config=push_cfg)
                    decision.stored_id = task_result.task_id
                    if self.config.step_trace_enabled:
                        trace_node(message_id=message.message_id, node_name="observe_logic3_check")
                    ferment_results = apply_logic3_on_task(
                        card=task_card,
                        state_store=self.state_store,
                        threshold=self.config.observe_ferment_threshold,
                        base_score=self.config.observe_logic3_base,
                    )
                    for item in ferment_results:
                        self._append_observe_ferment_event(message.message_id, task_card.card_id, item.to_dict())
                    self._process_observe_pop(message=message, result=result)
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
                    answered_this_decision = False
                    should_question = is_question_with_llm(summary=card.summary, problem=card.problem, content=message.content_text)
                    if should_question:
                        result.observe_question_count += 1
                    if should_question and self.config.observe_auto_reply_enabled and self.config.rag_enabled:
                        if self.config.step_trace_enabled:
                            trace_node(message_id=message.message_id, node_name="rag_retrieve_observe")
                        observe_query = f"{card.summary} {card.problem} {message.content_text}".strip()
                        try:
                            hits = retrieve(
                                self.vector_store,
                                query=observe_query,
                                top_k=self.config.rag_top_k,
                                min_score=self.config.rag_min_score,
                            )
                        except Exception as exc:
                            hits = []
                            result.warnings.append(f"observe rag retrieval failed: {exc}")
                        if hits:
                            result.rag_retrieval_count += 1
                        answer_result = try_answer_with_rag(observe_query, hits)
                        if answer_result.can_answer:
                            if self.config.step_trace_enabled:
                                trace_node(message_id=message.message_id, node_name="observe_reply")
                            sent = push_text_message(chat_id=message.chat_id, text=answer_result.answer, env_file=self.config.env_file)
                            self._append_observe_reply_event(
                                ObserveReplyEvent(
                                    message_id=message.message_id,
                                    chat_id=message.chat_id,
                                    query=observe_query,
                                    status=sent.status,
                                    answer=answer_result.answer,
                                    error=sent.error,
                                    hit_count=len(answer_result.hits or []),
                                )
                            )
                            if sent.status == "sent":
                                result.observe_answered_count += 1
                                answered_this_decision = True
                            else:
                                result.errors.append(sent.error)
                                trace_status = "failed"
                        else:
                            result.observe_fallback_count += 1
                    if (not answered_this_decision) or (not should_question):
                        if should_question and not answered_this_decision:
                            result.observe_fallback_count += 1
                        if self.config.step_trace_enabled:
                            trace_node(message_id=message.message_id, node_name="observe_store")
                        decision.stored_id = save_observe(self.state_store, card)
                        if self.config.step_trace_enabled:
                            trace_node(message_id=message.message_id, node_name="observe_logic1_check")
                        ferment_result = apply_logic1_on_observe_add(
                            card=card,
                            message=message,
                            state_store=self.state_store,
                            threshold=self.config.observe_ferment_threshold,
                            base_score=self.config.observe_logic1_base,
                        )
                        if ferment_result is not None:
                            self._append_observe_ferment_event(message.message_id, card.card_id, ferment_result.to_dict())
                        self._process_observe_pop(message=message, result=result)
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

    def _append_observe_reply_event(self, event: ObserveReplyEvent) -> None:
        append_jsonl(self.observe_reply_events_path, [event.to_dict()])

    def _append_observe_ferment_event(self, message_id: str, card_id: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.observe_ferment_events_path,
            [
                {
                    "message_id": message_id,
                    "card_id": card_id,
                    **payload,
                    "created_at": now_utc_iso(),
                }
            ],
        )

    def _append_observe_pop_event(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.observe_pop_events_path, [{**payload, "created_at": now_utc_iso()}])

    def _process_observe_pop(self, *, message: MessageEvent, result: EngineResult) -> None:
        ready_items = pop_ready_items(self.state_store, threshold=self.config.observe_ferment_threshold)
        for item in ready_items:
            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="observe_pop_route")
            observe_card = self._build_observe_card(item)
            decisions = route_cards(
                [observe_card],
                knowledge_threshold=self.config.knowledge_threshold,
                task_threshold=self.config.task_threshold,
            )
            if not decisions:
                continue
            decision = decisions[0]
            reroute_target = decision.target_pool
            final_target = reroute_target
            if reroute_target == "observe" and self.config.observe_force_non_observe_on_pop:
                final_target = "task" if observe_card.confidence >= self.config.task_threshold else "knowledge"
            if final_target == "task":
                task_result = save_task(
                    store=self.state_store,
                    card=observe_card,
                    run_id=self._build_run_id(message),
                    push_config=TaskPushConfig(enabled=False, chat_id="", env_file=self.config.env_file),
                )
                stored_id = task_result.task_id
                result.observe_reroute_task_count += 1
            else:
                stored_id = save_knowledge(self.state_store, observe_card, vector_store=self.vector_store)
                result.observe_reroute_knowledge_count += 1
            self.state_store.mark_observe_popped(str(item.get("observe_id", "")), final_target=final_target)
            pop_item = ObservePopItem(
                observe_id=str(item.get("observe_id", "")),
                topic=str(item.get("topic", "")),
                ferment_score=float(item.get("ferment_score", 0.0) or 0.0),
                reroute_target=reroute_target,
                final_target=final_target,
                reason_codes=list(decision.reason_codes),
            )
            self._append_observe_pop_event(
                {
                    **pop_item.to_dict(),
                    "stored_id": stored_id,
                    "message_id": message.message_id,
                    "chat_id": message.chat_id,
                }
            )
            result.observe_pop_count += 1

    @staticmethod
    def _build_observe_card(item: dict[str, Any]) -> LiftedCard:
        observe_id = str(item.get("observe_id", "obs"))
        topic = str(item.get("topic", "")).strip() or "observe topic"
        evidence = [str(v) for v in item.get("evidence", []) if str(v).strip()] if isinstance(item.get("evidence"), list) else []
        confidence = min(0.95, max(0.5, float(item.get("ferment_score", 0.0) or 0.0) / 5.0))
        return LiftedCard(
            card_id=f"pop-{observe_id}",
            candidate_id=f"pop-{observe_id}",
            title=topic,
            summary=f"Observe popped: {topic}",
            problem="Observe fermentation threshold reached.",
            suggestion="Please route to executable action or knowledge.",
            target_audience="team",
            evidence=evidence[:5],
            tags=["observe-pop"],
            confidence=confidence,
            suggested_target="observe",
            source_message_ids=[],
        )

    @staticmethod
    def _build_run_id(message: MessageEvent) -> str:
        return f"rt-{message.message_id}"


__all__ = ["Engine", "EngineConfig", "EngineResult"]
