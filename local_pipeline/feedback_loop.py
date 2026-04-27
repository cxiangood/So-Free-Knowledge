from __future__ import annotations

import hashlib
from typing import Any

from .shared_types import now_utc_iso


def generate_secondary_candidates(
    state_snapshot: dict[str, list[dict[str, Any]]],
    *,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tasks = state_snapshot.get("tasks", [])
    observe = state_snapshot.get("observe", [])

    unfinished = [item for item in tasks if str(item.get("status", "")) not in {"done", "dropped"}]
    for task in unfinished[: top_k // 2]:
        action = str(task.get("action", "")).strip()
        if not action:
            continue
        text = f"任务长期未完成: {action}"
        out.append(
            {
                "candidate_id": "sec-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12],
                "source": "unfinished_task",
                "content": text,
                "score_hint": 0.62,
                "created_at": now_utc_iso(),
            }
        )

    hot_observations = sorted(
        [item for item in observe if int(item.get("hit_count", 0)) >= 2],
        key=lambda x: int(x.get("hit_count", 0)),
        reverse=True,
    )
    for item in hot_observations[: top_k - len(out)]:
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        text = f"重复弱信号持续出现: {topic}"
        out.append(
            {
                "candidate_id": "sec-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12],
                "source": "repeated_observe",
                "content": text,
                "score_hint": 0.58,
                "created_at": now_utc_iso(),
            }
        )
    return out

