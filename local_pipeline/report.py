from __future__ import annotations

from typing import Any


def build_run_metrics(
    *,
    run_id: str,
    message_count: int,
    candidate_count: int,
    route_counts: dict[str, int],
    push_count: int,
    feedback_summary: dict[str, Any],
    secondary_count: int,
) -> dict[str, Any]:
    knowledge_count = int(route_counts.get("knowledge", 0))
    task_count = int(route_counts.get("task", 0))
    observe_count = int(route_counts.get("observe", 0))
    captured = float(candidate_count) / float(message_count) if message_count else 0.0
    upgraded = float(knowledge_count + task_count) / float(candidate_count) if candidate_count else 0.0
    route_total = max(1, knowledge_count + task_count + observe_count)
    done = int(feedback_summary.get("done_count", 0))
    pending = int(feedback_summary.get("pending_count", 0))
    blocked = int(feedback_summary.get("blocked_count", 0))
    completion = float(done) / float(max(1, done + pending + blocked))

    return {
        "run_id": run_id,
        "message_count": message_count,
        "candidate_count": candidate_count,
        "capture_rate": round(captured, 4),
        "upgrade_rate": round(upgraded, 4),
        "route_counts": {
            "knowledge": knowledge_count,
            "task": task_count,
            "observe": observe_count,
        },
        "route_ratio": {
            "knowledge": round(knowledge_count / route_total, 4),
            "task": round(task_count / route_total, 4),
            "observe": round(observe_count / route_total, 4),
        },
        "push_count": push_count,
        "feedback": feedback_summary,
        "closure_conversion_rate": round(completion, 4),
        "secondary_candidate_count": secondary_count,
    }


def render_markdown_report(metrics: dict[str, Any]) -> str:
    route_counts = metrics.get("route_counts", {})
    route_ratio = metrics.get("route_ratio", {})
    feedback = metrics.get("feedback", {})
    lines = [
        f"# Local Pipeline Run Report - {metrics.get('run_id', '')}",
        "",
        "## Summary",
        f"- Message count: {metrics.get('message_count', 0)}",
        f"- Candidate count: {metrics.get('candidate_count', 0)}",
        f"- Capture rate: {metrics.get('capture_rate', 0)}",
        f"- Upgrade rate: {metrics.get('upgrade_rate', 0)}",
        f"- Push events: {metrics.get('push_count', 0)}",
        "",
        "## Routing",
        f"- knowledge: {route_counts.get('knowledge', 0)} ({route_ratio.get('knowledge', 0)})",
        f"- task: {route_counts.get('task', 0)} ({route_ratio.get('task', 0)})",
        f"- observe: {route_counts.get('observe', 0)} ({route_ratio.get('observe', 0)})",
        "",
        "## Feedback",
        f"- updated: {feedback.get('updated_count', 0)}",
        f"- done: {feedback.get('done_count', 0)}",
        f"- delayed: {feedback.get('delayed_count', 0)}",
        f"- blocked: {feedback.get('blocked_count', 0)}",
        f"- closure_conversion_rate: {metrics.get('closure_conversion_rate', 0)}",
        "",
        "## Secondary Inspiration",
        f"- secondary_candidate_count: {metrics.get('secondary_candidate_count', 0)}",
    ]
    return "\n".join(lines).strip() + "\n"

