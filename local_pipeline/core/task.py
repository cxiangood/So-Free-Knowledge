from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from ..comm.identity_map import UserIdentityMap, parse_participants_names
from ..comm.send import TaskPushAttempt, TaskPushConfig, push_task_card, push_task_card_to_user_targets
from ..shared.models import LiftedCard, RagHit
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
    source_chat_id: str = "",
    identity_map: UserIdentityMap | None = None,
) -> TaskHandleResult:
    task_id = store.add_task(card)
    attempt: TaskPushAttempt | None = None
    if push_config.enabled:
        names = parse_participants_names(card.participants)
        if identity_map is not None and names and source_chat_id:
            targets: list[tuple[str, str, str]] = []
            fallback_id = str(os.getenv("TASK_PUSH_FALLBACK_RECEIVE_ID", "")).strip()
            fallback_id_type = str(os.getenv("TASK_PUSH_FALLBACK_RECEIVE_ID_TYPE", "")).strip() or str(
                os.getenv("TASK_PUSH_RECEIVE_ID_TYPE", "user_id")
            ).strip()
            for name in names:
                rid, err = identity_map.resolve_name_in_chat(chat_id=source_chat_id, name=name)
                if rid:
                    targets.append((name, rid, ""))
                else:
                    targets.append((name, "", err or "resolve_failed"))
            if targets:
                attempt = push_task_card_to_user_targets(
                    run_id=run_id,
                    task_id=task_id,
                    card=card,
                    env_file=push_config.env_file,
                    targets=targets,
                    fallback_receive_id=fallback_id,
                    fallback_receive_id_type=fallback_id_type,
                )
            else:
                attempt = TaskPushAttempt(
                    task_id=task_id,
                    run_id=run_id,
                    chat_id=source_chat_id,
                    card_payload={},
                    status="failed",
                    error="no_resolvable_target_in_participants",
                    details=[],
                )
        else:
            attempt = push_task_card(config=push_config, run_id=run_id, task_id=task_id, card=card)
    return TaskHandleResult(task_id=task_id, push_attempt=attempt)


def enhance_task_card_with_rag(card: LiftedCard, hits: list[RagHit], max_hits: int = 3) -> LiftedCard:
    if not hits:
        return card
    top = hits[: max(1, int(max_hits or 1))]
    refs = [f"{idx+1}. {item.title or item.summary} (score={item.score:.3f})" for idx, item in enumerate(top)]
    ref_text = " | ".join(refs)
    new_summary = _clip(f"{card.summary} | Related: {ref_text}", 220)
    new_suggestion = _clip(f"{card.suggestion} | Refer knowledge before execution.", 200)
    new_evidence = list(card.evidence)
    for item in top:
        snippet = item.summary or item.text
        if snippet and snippet not in new_evidence:
            new_evidence.append(_clip(snippet, 180))
    new_evidence = new_evidence[:8]
    return LiftedCard(
        card_id=card.card_id,
        candidate_id=card.candidate_id,
        title=card.title,
        summary=new_summary,
        problem=card.problem,
        suggestion=new_suggestion,
        participants=list(card.participants),
        times=str(card.times or ""),
        locations=str(card.locations or ""),
        evidence=new_evidence,
        tags=card.tags,
        confidence=card.confidence,
        suggested_target=card.suggested_target,
        source_message_ids=card.source_message_ids,
        topic_focus=card.topic_focus,
        message_role=card.message_role,
        context_relation=card.context_relation,
        context_evidence=list(card.context_evidence),
        decision_signals=dict(card.decision_signals),
        missing_fields=list(card.missing_fields),
    )


def _clip(text: str, max_len: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


__all__ = ["TaskHandleResult", "save_task", "enhance_task_card_with_rag"]
