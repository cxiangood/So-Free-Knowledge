from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..msg.parse import load_plain_messages_from_archive, plain_message_to_event
from .engine import Engine, EngineConfig


DEFAULT_MESSAGES_FILE = Path("message_archive/20260427T043305Z/messages.jsonl")


@dataclass(slots=True)
class OfflineConfig:
    messages_file: str | Path = DEFAULT_MESSAGES_FILE
    output_dir: str | Path = "outputs/local_pipeline"
    state_dir: str | Path = "outputs/local_pipeline/state"
    chat_history_path: str | Path = "outputs/local_pipeline/state/chat_message_store.json"
    chat_history_limit: int = 100
    context_window_size: int = 20
    detect_threshold: float = 45.0
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


def run(config: OfflineConfig | None = None) -> dict[str, Any]:
    cfg = config or OfflineConfig()
    engine = Engine(
        EngineConfig(
            output_dir=cfg.output_dir,
            state_dir=cfg.state_dir,
            chat_history_path=cfg.chat_history_path,
            chat_history_limit=cfg.chat_history_limit,
            context_window_size=cfg.context_window_size,
            detect_threshold=cfg.detect_threshold,
            task_push_enabled=cfg.task_push_enabled,
            task_push_chat_id=cfg.task_push_chat_id,
            env_file=cfg.env_file,
            step_trace_enabled=cfg.step_trace_enabled,
            rag_enabled=cfg.rag_enabled,
            rag_top_k=cfg.rag_top_k,
            rag_min_score=cfg.rag_min_score,
            rag_embed_model=cfg.rag_embed_model,
            observe_auto_reply_enabled=cfg.observe_auto_reply_enabled,
            observe_ferment_threshold=cfg.observe_ferment_threshold,
            observe_logic1_base=cfg.observe_logic1_base,
            observe_logic2_base=cfg.observe_logic2_base,
            observe_logic3_base=cfg.observe_logic3_base,
            observe_force_non_observe_on_pop=cfg.observe_force_non_observe_on_pop,
        )
    )

    messages = load_plain_messages_from_archive(cfg.messages_file)
    total = 0
    candidate_total = 0
    push_attempted = 0
    push_sent = 0
    push_failed = 0
    route_counts: dict[str, int] = {}
    warnings: list[str] = []
    errors: list[str] = []
    rag_retrieval_count = 0
    observe_question_count = 0
    observe_answered_count = 0
    observe_fallback_count = 0
    observe_pop_count = 0
    observe_reroute_task_count = 0
    observe_reroute_knowledge_count = 0
    denoise_filtered_count = 0

    for msg in messages:
        evt = plain_message_to_event(msg)
        result = engine.run(evt, context={"mode": "offline"})
        total += 1
        candidate_total += int(result.candidate_count)
        push_attempted += int(result.task_push_attempted)
        push_sent += int(result.task_push_sent)
        push_failed += int(result.task_push_failed)
        for key, value in result.routed_counts.items():
            route_counts[key] = route_counts.get(key, 0) + int(value)
        warnings.extend(result.warnings)
        errors.extend(result.errors)
        rag_retrieval_count += int(result.rag_retrieval_count)
        observe_question_count += int(result.observe_question_count)
        observe_answered_count += int(result.observe_answered_count)
        observe_fallback_count += int(result.observe_fallback_count)
        observe_pop_count += int(result.observe_pop_count)
        observe_reroute_task_count += int(result.observe_reroute_task_count)
        observe_reroute_knowledge_count += int(result.observe_reroute_knowledge_count)
        denoise_filtered_count += int(result.denoise_filtered_count)

    return {
        "ok": True,
        "mode": "offline",
        "messages_file": str(cfg.messages_file),
        "message_count": total,
        "candidate_count": candidate_total,
        "route_counts": route_counts,
        "task_push_attempted": push_attempted,
        "task_push_sent": push_sent,
        "task_push_failed": push_failed,
        "rag_retrieval_count": rag_retrieval_count,
        "observe_question_count": observe_question_count,
        "observe_answered_count": observe_answered_count,
        "observe_fallback_count": observe_fallback_count,
        "observe_pop_count": observe_pop_count,
        "observe_reroute_task_count": observe_reroute_task_count,
        "observe_reroute_knowledge_count": observe_reroute_knowledge_count,
        "denoise_filtered_count": denoise_filtered_count,
        "warnings": warnings,
        "errors": errors,
    }


__all__ = ["DEFAULT_MESSAGES_FILE", "OfflineConfig", "run"]
