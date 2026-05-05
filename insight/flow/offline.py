from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import get_config_bool, get_config_float, get_config_int, get_config_str

from ..msg.parse import load_plain_messages_from_archive, plain_message_to_event
from .engine import Engine, EngineConfig


DEFAULT_MESSAGES_FILE = Path(get_config_str("insight.offline.messages_file"))


@dataclass(slots=True)
class OfflineConfig:
    messages_file: str | Path = DEFAULT_MESSAGES_FILE
    output_dir: str | Path = get_config_str("insight.output_dir")
    state_dir: str | Path = get_config_str("insight.state_dir")
    chat_history_path: str | Path = get_config_str("insight.chat_history_path")
    chat_history_limit: int = get_config_int("insight.chat_history_limit")
    context_window_size: int = get_config_int("insight.context_window_size")
    detect_threshold: float = get_config_float("insight.detect_threshold")
    task_push_enabled: bool = get_config_bool("insight.task_push_enabled")
    env_file: str = get_config_str("insight.env_file")
    step_trace_enabled: bool = get_config_bool("insight.step_trace_enabled")
    rag_enabled: bool = get_config_bool("insight.rag_enabled")
    rag_top_k: int = get_config_int("insight.rag_top_k")
    rag_min_score: float = get_config_float("insight.rag_min_score")
    rag_embed_model: str = get_config_str("insight.rag_embed_model")
    observe_auto_reply_enabled: bool = get_config_bool("insight.observe_auto_reply_enabled")
    observe_ferment_threshold: float = get_config_float("insight.observe_ferment_threshold")
    observe_logic1_base: float = get_config_float("insight.observe_logic1_base")
    observe_logic2_base: float = get_config_float("insight.observe_logic2_base")
    observe_logic3_base: float = get_config_float("insight.observe_logic3_base")
    observe_force_non_observe_on_pop: bool = get_config_bool("insight.observe_force_non_observe_on_pop")


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
