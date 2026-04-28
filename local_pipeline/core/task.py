from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..comm.send import TaskPushAttempt, TaskPushConfig, push_task_card
from ..shared.models import LiftedCard
from ..store.state import LocalStateStore


@dataclass(slots=True)
class TaskHandleResult:
    task_id: str
    push_attempt: TaskPushAttempt | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "push_attempt": self.push_attempt.to_dict() if self.push_attempt else None,
        }


def save_task(
    *,
    store: LocalStateStore,
    card: LiftedCard,
    run_id: str,
    push_config: TaskPushConfig,
) -> TaskHandleResult:
    task_id = store.add_task(card)
    attempt: TaskPushAttempt | None = None
    if push_config.enabled:
        attempt = push_task_card(config=push_config, run_id=run_id, task_id=task_id, card=card)
    return TaskHandleResult(task_id=task_id, push_attempt=attempt)


__all__ = ["TaskHandleResult", "save_task"]
