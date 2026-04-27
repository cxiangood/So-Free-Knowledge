from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .io_utils import read_json
from .pipeline import PipelineConfig, run_pipeline
from .report import render_markdown_report
from .stores import LocalStateStore


def _parse_bool(text: str) -> bool:
    value = str(text or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def cmd_run(args: argparse.Namespace) -> int:
    config = PipelineConfig(
        output_dir=args.output_dir,
        state_dir=args.state_dir,
        observe_escalation_threshold=args.observe_escalation_threshold,
        candidate_threshold=args.candidate_threshold,
        knowledge_threshold=args.knowledge_threshold,
        task_threshold=args.task_threshold,
        task_push_enabled=_parse_bool(args.task_push_enabled),
        task_push_chat_id=str(args.task_push_chat_id or os.getenv("TASK_PUSH_CHAT_ID", "")).strip(),
        env_file=str(args.env_file or ""),
    )
    result = run_pipeline(
        messages_file=args.messages_file,
        run_id=args.run_id,
        enable_llm=_parse_bool(args.enable_llm),
        task_updates_file=args.task_updates_file,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    config = PipelineConfig(output_dir=args.output_dir, state_dir=args.state_dir, candidate_threshold=args.candidate_threshold)
    result = run_pipeline(
        messages_file=args.messages_file,
        run_id=args.run_id,
        enable_llm=False,
        task_updates_file=None,
        config=config,
    )
    summary = {
        "run_id": result["run_id"],
        "message_count": result["message_count"],
        "candidate_count": result["candidate_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    config = PipelineConfig(
        output_dir=args.output_dir,
        state_dir=args.state_dir,
        candidate_threshold=args.candidate_threshold,
        knowledge_threshold=args.knowledge_threshold,
        task_threshold=args.task_threshold,
    )
    result = run_pipeline(
        messages_file=args.messages_file,
        run_id=args.run_id,
        enable_llm=_parse_bool(args.enable_llm),
        task_updates_file=None,
        config=config,
    )
    summary = {
        "run_id": result["run_id"],
        "route_counts": result["route_counts"],
        "push_count": result["push_count"],
        "warnings": result.get("warnings", []),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    store = LocalStateStore(args.state_dir)
    summary = store.apply_task_updates(args.task_updates_file).to_dict()
    snapshot = store.snapshot()
    payload = {
        "feedback_summary": summary,
        "task_count": len(snapshot["tasks"]),
        "observe_count": len(snapshot["observe"]),
        "knowledge_count": len(snapshot["knowledge"]),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    metrics = read_json(run_dir / "run_metrics.json", {})
    if not isinstance(metrics, dict) or not metrics:
        raise FileNotFoundError(f"run_metrics.json not found or invalid under: {run_dir}")
    print(render_markdown_report(metrics))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-pipeline", description="Local closed-loop simulation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run full local closed-loop pipeline")
    run_parser.add_argument("--messages-file", required=True)
    run_parser.add_argument("--enable-llm", default="false")
    run_parser.add_argument("--env-file", default="")
    run_parser.add_argument("--run-id", default="")
    run_parser.add_argument("--output-dir", default="outputs/local_pipeline")
    run_parser.add_argument("--state-dir", default="outputs/local_pipeline/state")
    run_parser.add_argument("--task-updates-file", default="")
    run_parser.add_argument("--task-push-enabled", default="false")
    run_parser.add_argument("--task-push-chat-id", default="")
    run_parser.add_argument("--observe-escalation-threshold", type=int, default=3)
    run_parser.add_argument("--candidate-threshold", type=float, default=0.45)
    run_parser.add_argument("--knowledge-threshold", type=float, default=0.60)
    run_parser.add_argument("--task-threshold", type=float, default=0.50)
    run_parser.set_defaults(func=cmd_run)

    score_parser = sub.add_parser("score", help="Run scoring phase quickly")
    score_parser.add_argument("--messages-file", required=True)
    score_parser.add_argument("--run-id", default="")
    score_parser.add_argument("--output-dir", default="outputs/local_pipeline")
    score_parser.add_argument("--state-dir", default="outputs/local_pipeline/state")
    score_parser.add_argument("--candidate-threshold", type=float, default=0.45)
    score_parser.set_defaults(func=cmd_score)

    route_parser = sub.add_parser("route", help="Run score + lift + route")
    route_parser.add_argument("--messages-file", required=True)
    route_parser.add_argument("--enable-llm", default="false")
    route_parser.add_argument("--run-id", default="")
    route_parser.add_argument("--output-dir", default="outputs/local_pipeline")
    route_parser.add_argument("--state-dir", default="outputs/local_pipeline/state")
    route_parser.add_argument("--candidate-threshold", type=float, default=0.45)
    route_parser.add_argument("--knowledge-threshold", type=float, default=0.60)
    route_parser.add_argument("--task-threshold", type=float, default=0.50)
    route_parser.set_defaults(func=cmd_route)

    simulate_parser = sub.add_parser("simulate", help="Apply feedback updates to local task state")
    simulate_parser.add_argument("--state-dir", default="outputs/local_pipeline/state")
    simulate_parser.add_argument("--task-updates-file", required=True)
    simulate_parser.set_defaults(func=cmd_simulate)

    report_parser = sub.add_parser("report", help="Render markdown report from a run directory")
    report_parser.add_argument("--run-dir", required=True)
    report_parser.set_defaults(func=cmd_report)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
