from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .feedback_loop import generate_secondary_candidates
from .ingest import ingest_messages
from .io_utils import append_jsonl, write_json, write_jsonl
from .report import build_run_metrics, render_markdown_report
from .router import route_cards
from .semantic_lifter import lift_candidates
from .signal_detector import detect_candidates
from .simulator import simulate_push_events
from .stores import LocalStateStore
from .task_card_sender import TaskPushConfig, push_task_card


@dataclass(slots=True)
class PipelineConfig:
    output_dir: str | Path = "outputs/local_pipeline"
    state_dir: str | Path = "outputs/local_pipeline/state"
    observe_escalation_threshold: int = 3
    candidate_threshold: float = 0.45
    knowledge_threshold: float = 0.60
    task_threshold: float = 0.50
    task_push_enabled: bool = False
    task_push_chat_id: str = ""
    env_file: str = ""


def _resolve_run_id(run_id: str = "") -> str:
    if run_id.strip():
        return run_id.strip()
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_pipeline(
    *,
    messages_file: str | Path,
    run_id: str = "",
    enable_llm: bool = False,
    task_updates_file: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PipelineConfig()
    final_run_id = _resolve_run_id(run_id)
    run_dir = Path(cfg.output_dir) / final_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    messages = ingest_messages(messages_file)
    detection = detect_candidates(messages, candidate_threshold=cfg.candidate_threshold)
    lift_result = lift_candidates(detection.candidates, detection.messages, enable_llm=enable_llm)
    decisions = route_cards(
        lift_result.cards,
        knowledge_threshold=cfg.knowledge_threshold,
        task_threshold=cfg.task_threshold,
    )

    store = LocalStateStore(cfg.state_dir)
    task_push_attempted = 0
    task_push_sent = 0
    task_push_failed = 0
    task_push_failures: list[dict[str, str]] = []
    pending_queue_rows: list[dict[str, Any]] = []
    task_push_attempts: list[dict[str, str]] = []
    task_push_config = TaskPushConfig(
        enabled=bool(cfg.task_push_enabled),
        chat_id=str(cfg.task_push_chat_id or ""),
        env_file=str(cfg.env_file or ""),
    )

    for decision in decisions:
        card = next((item for item in lift_result.cards if item.card_id == decision.card_id), None)
        if card is None:
            continue
        if decision.target_pool == "knowledge":
            decision.stored_id = store.add_knowledge(card)
        elif decision.target_pool == "task":
            decision.stored_id = store.add_task(card)
            if task_push_config.enabled:
                task_push_attempted += 1
                attempt = push_task_card(
                    config=task_push_config,
                    run_id=final_run_id,
                    task_id=decision.stored_id,
                    card=card,
                )
                task_push_attempts.append(
                    {
                        "task_id": attempt.task_id,
                        "status": attempt.status,
                        "message_id": attempt.message_id,
                        "error": attempt.error,
                    }
                )
                if attempt.status == "sent":
                    task_push_sent += 1
                elif attempt.status == "failed":
                    task_push_failed += 1
                    task_push_failures.append({"task_id": decision.stored_id, "error": attempt.error})
                    pending_queue_rows.append(attempt.to_pending_dict())
        else:
            decision.stored_id = store.add_observe(card)

    escalations = store.escalate_observations(threshold=cfg.observe_escalation_threshold)
    push_events = simulate_push_events(final_run_id, decisions, escalations=escalations)
    task_delivery_by_task_id: dict[str, dict[str, str]] = {}
    for item in task_push_attempts:
        task_id = str(item.get("task_id", "")).strip()
        if not task_id:
            continue
        status = str(item.get("status", "")).strip()
        if status == "sent":
            task_delivery_by_task_id[task_id] = {
                "delivery_status": "sent",
                "message_id": str(item.get("message_id", "")),
                "error": "",
            }
        elif status == "failed":
            task_delivery_by_task_id[task_id] = {
                "delivery_status": "failed",
                "message_id": "",
                "error": str(item.get("error", "")),
            }
        else:
            task_delivery_by_task_id[task_id] = {
                "delivery_status": "skipped",
                "message_id": "",
                "error": str(item.get("error", "")),
            }

    for event in push_events:
        if event.target != "task":
            continue
        task_id = str(event.payload_ref or "")
        event.task_id = task_id
        delivery = task_delivery_by_task_id.get(task_id)
        if not delivery:
            event.delivery_status = "not_attempted"
            continue
        event.delivery_status = delivery.get("delivery_status", "not_attempted")
        event.message_id = delivery.get("message_id", "")
        event.error = delivery.get("error", "")

    if pending_queue_rows:
        append_jsonl(Path(cfg.state_dir) / "pending_task_push.jsonl", pending_queue_rows)

    feedback_summary = store.apply_task_updates(task_updates_file).to_dict()
    snapshot = store.snapshot()
    secondary = generate_secondary_candidates(snapshot)

    route_counts = Counter(item.target_pool for item in decisions)
    metrics = build_run_metrics(
        run_id=final_run_id,
        message_count=len(detection.messages),
        candidate_count=len(detection.candidates),
        route_counts=dict(route_counts),
        push_count=len(push_events),
        feedback_summary=feedback_summary,
        secondary_count=len(secondary),
    )
    if lift_result.warnings:
        metrics["warnings"] = lift_result.warnings
    store.append_run_metrics(metrics)

    write_jsonl(run_dir / "messages.jsonl", [item.to_dict() for item in detection.messages])
    write_jsonl(run_dir / "candidates.jsonl", [item.to_dict() for item in detection.candidates])
    write_jsonl(run_dir / "lifted_cards.jsonl", [item.to_dict() for item in lift_result.cards])
    write_jsonl(run_dir / "route_decisions.jsonl", [item.to_dict() for item in decisions])
    write_jsonl(run_dir / "push_events.jsonl", [item.to_dict() for item in push_events])
    write_jsonl(run_dir / "secondary_candidates.jsonl", secondary)
    write_json(run_dir / "feedback_summary.json", feedback_summary)
    write_json(run_dir / "run_metrics.json", metrics)
    (run_dir / "run_report.md").write_text(render_markdown_report(metrics), encoding="utf-8")

    return {
        "ok": True,
        "run_id": final_run_id,
        "run_dir": str(run_dir),
        "message_count": len(detection.messages),
        "candidate_count": len(detection.candidates),
        "route_counts": dict(route_counts),
        "push_count": len(push_events),
        "secondary_candidate_count": len(secondary),
        "warnings": lift_result.warnings,
        "task_push_attempted": task_push_attempted,
        "task_push_sent": task_push_sent,
        "task_push_failed": task_push_failed,
        "task_push_failures": task_push_failures,
    }
