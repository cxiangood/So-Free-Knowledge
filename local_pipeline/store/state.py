from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from ..shared.models import LiftedCard
from ..shared.utils import now_utc_iso
from .io import read_json, read_json_or_jsonl, write_json


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class FeedbackSummary:
    updated_count: int = 0
    done_count: int = 0
    delayed_count: int = 0
    pending_count: int = 0
    blocked_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalStateStore:
    def __init__(self, state_dir: str | Path) -> None:
        self._lock = RLock()
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.knowledge_path = self.root / "knowledge_store.json"
        self.task_path = self.root / "task_store.json"
        self.observe_path = self.root / "observe_store.json"
        self.metrics_path = self.root / "run_metrics.json"
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        if not self.knowledge_path.exists():
            write_json(self.knowledge_path, {"items": []})
        if not self.task_path.exists():
            write_json(self.task_path, {"items": []})
        if not self.observe_path.exists():
            write_json(self.observe_path, {"items": []})
        if not self.metrics_path.exists():
            write_json(self.metrics_path, {"runs": []})

    def _load_items(self, path: Path) -> list[dict[str, Any]]:
        data = read_json(path, {"items": []})
        if not isinstance(data, dict):
            return []
        items = data.get("items", [])
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    def _save_items(self, path: Path, items: list[dict[str, Any]]) -> None:
        write_json(path, {"items": items})

    def add_knowledge(self, card: LiftedCard) -> str:
        with self._lock:
            items = self._load_items(self.knowledge_path)
            entry_id = "kg-" + hashlib.md5(card.card_id.encode("utf-8")).hexdigest()[:12]
            items.append(
                {
                    "card_id": card.card_id,
                    "card_ref": card.to_dict(),
                    "knowledge_id": entry_id,
                    "title": card.title,
                    "summary": card.summary,
                    "tags": card.tags,
                    "confidence": card.confidence,
                    "created_at": now_utc_iso(),
                }
            )
            self._save_items(self.knowledge_path, items)
            return entry_id

    def add_task(self, card: LiftedCard) -> str:
        with self._lock:
            items = self._load_items(self.task_path)
            task_id = "task-" + hashlib.md5(card.card_id.encode("utf-8")).hexdigest()[:12]
            items.append(
                {
                    "card_id": card.card_id,
                    "task_id": task_id,
                    "action": card.suggestion,
                    "owner_hint": card.target_audience,
                    "deadline_hint": "",
                    "priority": _priority_from_confidence(card.confidence),
                    "evidence": card.evidence,
                    "status": "todo",
                    "created_at": now_utc_iso(),
                    "updated_at": now_utc_iso(),
                    "history": [],
                }
            )
            self._save_items(self.task_path, items)
            return task_id

    def add_observe(self, card: LiftedCard) -> str:
        with self._lock:
            items = self._load_items(self.observe_path)
            topic = card.title.strip().lower()
            observe_id = "obs-" + hashlib.md5(topic.encode("utf-8")).hexdigest()[:12]
            matched = next((item for item in items if str(item.get("observe_id", "")) == observe_id), None)
            if matched is None:
                matched = {
                    "observe_id": observe_id,
                    "topic": card.title,
                    "card_ref": card.to_dict(),
                    "hit_count": 0,
                    "last_seen_at": "",
                    "escalation_state": "tracking",
                    "evidence": [],
                    "ferment_score": 0.0,
                    "logic1_count": 0,
                    "logic2_count": 0,
                    "logic3_count": 0,
                    "last_logic": "",
                    "last_trigger_at": "",
                    "pop_status": "tracking",
                    "pop_count": 0,
                    "last_popped_at": "",
                }
                items.append(matched)
            else:
                # Keep the latest lifted full card snapshot for observe replay/escalation.
                matched["card_ref"] = card.to_dict()
            matched["hit_count"] = int(matched.get("hit_count", 0)) + 1
            matched["last_seen_at"] = now_utc_iso()
            evidence = matched.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []
            for snippet in card.evidence:
                if snippet not in evidence:
                    evidence.append(snippet)
            matched["evidence"] = evidence[:10]
            self._save_items(self.observe_path, items)
            return observe_id

    def get_observe_item(self, observe_id: str) -> dict[str, Any] | None:
        items = self._load_items(self.observe_path)
        normalized = str(observe_id or "").strip()
        if not normalized:
            return None
        return next((item for item in items if str(item.get("observe_id", "")) == normalized), None)

    def list_observe_items(self) -> list[dict[str, Any]]:
        return self._load_items(self.observe_path)

    def apply_observe_ferment(self, observe_id: str, *, logic: str, score_added: float) -> dict[str, Any] | None:
        with self._lock:
            items = self._load_items(self.observe_path)
            normalized = str(observe_id or "").strip()
            if not normalized:
                return None
            matched = next((item for item in items if str(item.get("observe_id", "")) == normalized), None)
            if matched is None:
                return None
            if str(matched.get("pop_status", "tracking")) == "popped":
                return matched
            before = float(matched.get("ferment_score", 0.0) or 0.0)
            matched["ferment_score"] = round(before + float(score_added), 4)
            if logic == "logic1":
                matched["logic1_count"] = int(matched.get("logic1_count", 0)) + 1
            elif logic == "logic2":
                matched["logic2_count"] = int(matched.get("logic2_count", 0)) + 1
            elif logic == "logic3":
                matched["logic3_count"] = int(matched.get("logic3_count", 0)) + 1
            matched["last_logic"] = logic
            matched["last_trigger_at"] = now_utc_iso()
            self._save_items(self.observe_path, items)
            return matched

    def mark_observe_popped(self, observe_id: str, *, final_target: str) -> dict[str, Any] | None:
        with self._lock:
            items = self._load_items(self.observe_path)
            normalized = str(observe_id or "").strip()
            if not normalized:
                return None
            matched = next((item for item in items if str(item.get("observe_id", "")) == normalized), None)
            if matched is None:
                return None
            matched["pop_status"] = "popped"
            matched["pop_count"] = int(matched.get("pop_count", 0)) + 1
            matched["last_popped_at"] = now_utc_iso()
            matched["escalation_state"] = f"popped_{final_target}"
            self._save_items(self.observe_path, items)
            return matched

    def escalate_observations(self, threshold: int = 3) -> list[dict[str, Any]]:
        with self._lock:
            items = self._load_items(self.observe_path)
            escalated: list[dict[str, Any]] = []
            for item in items:
                hit = int(item.get("hit_count", 0))
                state = str(item.get("escalation_state", "tracking"))
                if hit < threshold or state != "tracking":
                    continue
                target = "task" if hit >= threshold + 1 else "knowledge"
                item["escalation_state"] = f"escalated_{target}"
                item["escalated_at"] = now_utc_iso()
                escalated.append(
                    {
                        "observe_id": item.get("observe_id", ""),
                        "topic": item.get("topic", ""),
                        "target": target,
                        "hit_count": hit,
                    }
                )
            if escalated:
                self._save_items(self.observe_path, items)
            return escalated

    def apply_task_updates(self, updates_file: str | Path | None) -> FeedbackSummary:
        summary = FeedbackSummary()
        if not updates_file:
            return summary
        path = Path(updates_file)
        if not path.exists():
            return summary
        updates = read_json_or_jsonl(path)
        tasks = self._load_items(self.task_path)
        by_id = {str(item.get("task_id", "")): item for item in tasks}
        for update in updates:
            task_id = str(update.get("task_id", "")).strip()
            if not task_id or task_id not in by_id:
                continue
            status = str(update.get("status", "")).strip().lower()
            if status not in {"todo", "in_progress", "done", "blocked", "dropped"}:
                continue
            task = by_id[task_id]
            task["status"] = status
            task["updated_at"] = str(update.get("updated_at", now_utc_iso()))
            comment = str(update.get("comment", "")).strip()
            history = task.get("history", [])
            if not isinstance(history, list):
                history = []
            history.append({"status": status, "comment": comment, "updated_at": task["updated_at"]})
            task["history"] = history[-50:]
            summary.updated_count += 1
        self._save_items(self.task_path, list(by_id.values()))
        self._fill_feedback_stats(summary)
        return summary

    def _fill_feedback_stats(self, summary: FeedbackSummary) -> None:
        tasks = self._load_items(self.task_path)
        now = _now_dt()
        for task in tasks:
            status = str(task.get("status", "todo"))
            if status == "done":
                summary.done_count += 1
            elif status == "blocked":
                summary.blocked_count += 1
            else:
                summary.pending_count += 1
            due_at = str(task.get("due_at", "")).strip()
            if due_at and status != "done":
                try:
                    due_dt = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if now > due_dt.astimezone(timezone.utc):
                    summary.delayed_count += 1

    def append_run_metrics(self, run_metrics: dict[str, Any]) -> None:
        with self._lock:
            data = read_json(self.metrics_path, {"runs": []})
            if not isinstance(data, dict):
                data = {"runs": []}
            runs = data.get("runs", [])
            if not isinstance(runs, list):
                runs = []
            runs.append(run_metrics)
            data["runs"] = runs[-200:]
            write_json(self.metrics_path, data)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "knowledge": self._load_items(self.knowledge_path),
            "tasks": self._load_items(self.task_path),
            "observe": self._load_items(self.observe_path),
        }


def _priority_from_confidence(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"

__all__ = ["FeedbackSummary", "LocalStateStore"]
