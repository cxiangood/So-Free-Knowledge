from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..comm.feishu import resolve_sender_credentials
from ..comm.identity_map import UserIdentityMap, resolve_identity_map_config
from ..comm.send import TaskPushAttempt, TaskPushConfig, push_text_message, queue_failed_pushes
from ..core.detect import detect_candidates
from ..core.kb import save_knowledge
from ..core.lift import lift_candidates
from ..core.obs import save_observe
from ..core.observe_ferment import apply_logic1_on_observe_add, apply_logic2_on_knowledge, apply_logic3_on_task, pop_ready_items
from ..core.observe_qa import is_question_with_llm, try_answer_with_rag
from ..core.route import route_cards
from ..core.task import enhance_task_card_with_rag, save_task
from ..msg.cache import ChatMessageStore
from ..msg.parse import event_row_to_message_event
from ..msg.types import MessageEvent
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
    detect_threshold: float = 40
    task_push_enabled: bool = False
    task_push_chat_id: str = ""
    env_file: str = ""
    step_trace_enabled: bool = True
    rag_enabled: bool = True
    rag_top_k: int = 5
    rag_min_score: float = 0.35
    rag_embed_model: str = "BAAI/bge-large-zh"
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


class EngineGraphState(TypedDict, total=False):
    message: MessageEvent
    result: EngineResult
    simple_messages: list[str]
    detect_score: float
    cards: list[LiftedCard]
    decisions: list[Any]
    decision_index: int
    failed_attempts: list[TaskPushAttempt]
    trace_status: str
    trace_started: bool


class RuntimeState:
    def __init__(self, path: str | Path, max_processed_ids: int = 2000) -> None:
        self.path = Path(path)
        self.max_processed_ids = max(100, int(max_processed_ids))
        self._lock = RLock()
        self._inflight: set[str] = set()

    def try_start(self, message_id: str) -> bool:
        normalized = str(message_id or "").strip()
        if not normalized:
            return False
        with self._lock:
            if normalized in self._inflight:
                return False
            data = read_json(self.path, {"processed_ids": []})
            if not isinstance(data, dict):
                data = {"processed_ids": []}
            ids = data.get("processed_ids", [])
            if not isinstance(ids, list):
                ids = []
            if normalized in ids:
                return False
            self._inflight.add(normalized)
            return True

    def finish(self, message_id: str, *, success: bool) -> None:
        normalized = str(message_id or "").strip()
        if not normalized:
            return
        with self._lock:
            self._inflight.discard(normalized)
            if not success:
                return
            data = read_json(self.path, {"processed_ids": []})
            if not isinstance(data, dict):
                data = {"processed_ids": []}
            ids = data.get("processed_ids", [])
            if not isinstance(ids, list):
                ids = []
            if normalized not in ids:
                ids.append(normalized)
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
        self.identity_map = UserIdentityMap(resolve_identity_map_config(state_dir=config.state_dir, env_file=config.env_file))
        if self.config.task_push_enabled:
            try:
                app_id, app_secret = resolve_sender_credentials()
                if app_id and app_secret:
                    self.identity_map.ensure_bootstrap(app_id=app_id, app_secret=app_secret)
            except Exception:
                pass
        if self.config.rag_enabled:
            try:
                self.vector_store = VectorKnowledgeStore(Path(config.state_dir) / "vector_kb", embed_model=self.config.rag_embed_model)
            except Exception:
                self.vector_store = None
        self._graph = self._build_graph()

    def run(self, message: MessageEvent, context: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> EngineResult:
        del context
        del config
        result = EngineResult(message_id=message.message_id, chat_id=message.chat_id)
        try:
            final_state = self._graph.invoke(
                {
                    "message": message,
                    "result": result,
                    "simple_messages": [],
                    "detect_score": 0.0,
                    "cards": [],
                    "decisions": [],
                    "decision_index": 0,
                    "failed_attempts": [],
                    "trace_status": "ok",
                    "trace_started": False,
                }
            )
        except Exception:
            self.runtime_state.finish(message.message_id, success=False)
            if self.config.step_trace_enabled:
                trace_finish(message_id=message.message_id, suffix_status="failed")
            raise
        return final_state["result"]

    def _build_graph(self):
        graph = StateGraph(EngineGraphState)
        graph.add_node("deduplicate", self._node_deduplicate)
        graph.add_node("message_cache", self._node_message_cache)
        graph.add_node("context_extract", self._node_context_extract)
        graph.add_node("signal_detect", self._node_signal_detect)
        graph.add_node("semantic_lift", self._node_semantic_lift)
        graph.add_node("route", self._node_route)
        graph.add_node("select_decision", self._node_select_decision)
        graph.add_node("knowledge_store", self._node_knowledge_store)
        graph.add_node("task_store", self._node_task_store)
        graph.add_node("observe_handle", self._node_observe_handle)
        graph.add_node("finalize", self._node_finalize)

        graph.set_entry_point("deduplicate")
        graph.add_conditional_edges("deduplicate", self._after_deduplicate, {"finalize": "finalize", "message_cache": "message_cache"})
        graph.add_edge("message_cache", "context_extract")
        graph.add_edge("context_extract", "signal_detect")
        graph.add_conditional_edges("signal_detect", self._after_signal_detect, {"finalize": "finalize", "semantic_lift": "semantic_lift"})
        graph.add_edge("semantic_lift", "route")
        graph.add_edge("route", "select_decision")
        graph.add_conditional_edges(
            "select_decision",
            self._next_decision_target,
            {
                "knowledge_store": "knowledge_store",
                "task_store": "task_store",
                "observe_handle": "observe_handle",
                "finalize": "finalize",
            },
        )
        graph.add_edge("knowledge_store", "select_decision")
        graph.add_edge("task_store", "select_decision")
        graph.add_edge("observe_handle", "select_decision")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _node_deduplicate(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        result = state["result"]
        if not self.runtime_state.try_start(message.message_id):
            result.skipped = True
            return {"result": result}
        if self.config.step_trace_enabled:
            trace_start(message_id=message.message_id, chat_id=message.chat_id, content=message.content_text)
            return {"trace_started": True}
        return {}

    @staticmethod
    def _after_deduplicate(state: EngineGraphState) -> str:
        return "finalize" if state["result"].skipped else "message_cache"

    def _node_message_cache(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        if self.config.step_trace_enabled:
            trace_node(message_id=message.message_id, node_name="message_cache")
        try:
            self.identity_map.update_from_event(message)
        except Exception:
            pass
        self.chat_store.append(message)
        return {}

    def _node_context_extract(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        if self.config.step_trace_enabled:
            trace_node(message_id=message.message_id, node_name="context_extract")
        context_rows = self.chat_store.get_chat_messages(message.chat_id)[-max(1, int(self.config.context_window_size)) :]
        message_events = [item for item in (event_row_to_message_event(row) for row in context_rows) if item is not None]
        return {"simple_messages": [item.get_simple_message() for item in message_events]}

    def _node_signal_detect(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        result = state["result"]
        if self.config.step_trace_enabled:
            trace_node(message_id=message.message_id, node_name="signal_detect")
        detection = detect_candidates(
            state.get("simple_messages", []),
            message_id=message.message_id,
            chat_id=message.chat_id,
        )
        detect_score = float(detection.value_score)
        result.candidate_count = 1 if detect_score >= float(self.config.detect_threshold) else 0
        return {"result": result, "detect_score": detect_score}

    def _after_signal_detect(self, state: EngineGraphState) -> str:
        score = float(state.get("detect_score", 0.0) or 0.0)
        return "semantic_lift" if score >= float(self.config.detect_threshold) else "finalize"

    def _node_semantic_lift(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        result = state["result"]
        if self.config.step_trace_enabled:
            trace_node(message_id=message.message_id, node_name="semantic_lift")
        lift_result = lift_candidates(state.get("simple_messages", []))
        result.warnings.extend(lift_result.warnings)
        return {"result": result, "cards": lift_result.cards}

    def _node_route(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        result = state["result"]
        if self.config.step_trace_enabled:
            trace_node(message_id=message.message_id, node_name="route")
        decisions = route_cards(
            state.get("cards", []),
            message_id=message.message_id,
            chat_id=message.chat_id,
        )
        for decision in decisions:
            result.routed_counts[decision.target_pool] = result.routed_counts.get(decision.target_pool, 0) + 1
        return {"result": result, "decisions": decisions, "decision_index": 0}

    @staticmethod
    def _node_select_decision(state: EngineGraphState) -> dict[str, Any]:
        del state
        return {}

    @staticmethod
    def _next_decision_target(state: EngineGraphState) -> str:
        decisions = state.get("decisions", [])
        index = int(state.get("decision_index", 0))
        if index >= len(decisions):
            return "finalize"
        target = decisions[index].target_pool
        if target == "knowledge":
            return "knowledge_store"
        if target == "task":
            return "task_store"
        return "observe_handle"

    def _current_decision_and_card(self, state: EngineGraphState) -> tuple[Any | None, LiftedCard | None]:
        decisions = state.get("decisions", [])
        index = int(state.get("decision_index", 0))
        if index >= len(decisions):
            return None, None
        decision = decisions[index]
        card = next((item for item in state.get("cards", []) if item.card_id == decision.card_id), None)
        return decision, card

    def _node_knowledge_store(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        result = state["result"]
        decision, card = self._current_decision_and_card(state)
        if decision is not None and card is not None:
            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="knowledge_store")
            decision.stored_id = save_knowledge(self.state_store, card, vector_store=self.vector_store)
            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="observe_logic2_check")
            for item in apply_logic2_on_knowledge(
                card=card,
                state_store=self.state_store,
                threshold=self.config.observe_ferment_threshold,
                base_score=self.config.observe_logic2_base,
            ):
                self._append_observe_ferment_event(message.message_id, card.card_id, item.to_dict())
            self._process_observe_pop(message=message, result=result)
        return {"result": result, "decision_index": int(state.get("decision_index", 0)) + 1}

    def _node_task_store(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        result = state["result"]
        failed_attempts = list(state.get("failed_attempts", []))
        trace_status = str(state.get("trace_status", "ok"))
        decision, card = self._current_decision_and_card(state)
        if decision is not None and card is not None:
            task_card = card
            if self.config.rag_enabled:
                if self.config.step_trace_enabled:
                    trace_node(message_id=message.message_id, node_name="rag_retrieve_task")
                query = f"{card.summary} {card.problem} {card.suggestion}".strip()
                try:
                    hits = retrieve(self.vector_store, query=query, top_k=self.config.rag_top_k, min_score=self.config.rag_min_score)
                except Exception as exc:
                    hits = []
                    result.warnings.append(f"task rag retrieval failed: {exc}")
                if hits:
                    result.rag_retrieval_count += 1
                    task_card = enhance_task_card_with_rag(card, hits)
            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="task_store")
            task_result = save_task(
                store=self.state_store,
                card=task_card,
                run_id=self._build_run_id(message),
                push_config=TaskPushConfig(enabled=self.config.task_push_enabled, chat_id=self.config.task_push_chat_id, env_file=self.config.env_file),
                source_chat_id=message.chat_id,
                identity_map=self.identity_map,
            )
            decision.stored_id = task_result.task_id
            if self.config.step_trace_enabled:
                trace_node(message_id=message.message_id, node_name="observe_logic3_check")
            for item in apply_logic3_on_task(
                card=task_card,
                state_store=self.state_store,
                threshold=self.config.observe_ferment_threshold,
                base_score=self.config.observe_logic3_base,
            ):
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
        return {
            "result": result,
            "failed_attempts": failed_attempts,
            "trace_status": trace_status,
            "decision_index": int(state.get("decision_index", 0)) + 1,
        }

    def _node_observe_handle(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        result = state["result"]
        trace_status = str(state.get("trace_status", "ok"))
        decision, card = self._current_decision_and_card(state)
        if decision is not None and card is not None:
            answered_this_decision = False
            should_question = is_question_with_llm(
                summary=card.summary,
                problem=card.problem,
                content=message.content_text,
                message_id=message.message_id,
                chat_id=message.chat_id,
            )
            if should_question:
                result.observe_question_count += 1
            if should_question and self.config.observe_auto_reply_enabled and self.config.rag_enabled:
                if self.config.step_trace_enabled:
                    trace_node(message_id=message.message_id, node_name="rag_retrieve_observe")
                observe_query = f"{card.summary} {card.problem} {message.content_text}".strip()
                try:
                    hits = retrieve(self.vector_store, query=observe_query, top_k=self.config.rag_top_k, min_score=self.config.rag_min_score)
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
        return {"result": result, "trace_status": trace_status, "decision_index": int(state.get("decision_index", 0)) + 1}

    def _node_finalize(self, state: EngineGraphState) -> dict[str, Any]:
        message = state["message"]
        result = state["result"]
        failed_attempts = list(state.get("failed_attempts", []))
        if failed_attempts:
            queue_failed_pushes(self.config.state_dir, failed_attempts)
        if not result.skipped:
            self.runtime_state.finish(message.message_id, success=True)
        self._append_result(result)
        if self.config.step_trace_enabled and state.get("trace_started"):
            if not result.skipped:
                trace_node(message_id=message.message_id, node_name="done")
            trace_finish(message_id=message.message_id, suffix_status=str(state.get("trace_status", "ok")))
        return {"result": result}

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
                message_id=message.message_id,
                chat_id=message.chat_id,
            )
            if not decisions:
                continue
            decision = next((item for item in decisions if item.target_pool != "observe"), decisions[0])
            reroute_target = decision.target_pool
            final_target = reroute_target
            if reroute_target == "observe" and self.config.observe_force_non_observe_on_pop:
                # Force non-observe by lifted semantics only (no score threshold fallback).
                final_target = "task" if observe_card.suggested_target == "task" else "knowledge"
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
            participants=["team"],
            times="",
            locations="",
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
