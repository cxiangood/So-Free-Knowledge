from __future__ import annotations

import hashlib

from .shared_types import PushEvent, RouteDecision


def simulate_push_events(
    run_id: str,
    decisions: list[RouteDecision],
    *,
    escalations: list[dict] | None = None,
) -> list[PushEvent]:
    events: list[PushEvent] = []
    for decision in decisions:
        if decision.target_pool == "observe":
            continue
        base = f"{run_id}:{decision.target_pool}:{decision.stored_id or decision.card_id}"
        event_id = "push-" + hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
        events.append(
            PushEvent(
                event_id=event_id,
                trigger_type="route_result",
                target=decision.target_pool,
                payload_ref=decision.stored_id or decision.card_id,
            )
        )

    for item in escalations or []:
        target = str(item.get("target", "task"))
        base = f"{run_id}:escalate:{item.get('observe_id','')}"
        event_id = "push-" + hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
        events.append(
            PushEvent(
                event_id=event_id,
                trigger_type="observe_threshold",
                target=target,
                payload_ref=str(item.get("observe_id", "")),
            )
        )
    return events

