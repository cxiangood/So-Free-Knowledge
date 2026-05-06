from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
import re

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
        knowledge_id, _ = self.add_or_merge_knowledge(card)
        return knowledge_id

    def add_or_merge_knowledge(self, card: LiftedCard, similarity_threshold: float = 0.72) -> tuple[str, bool]:
        with self._lock:
            items = self._load_items(self.knowledge_path)
            incoming_text = _semantic_text(card)
            matched = _find_most_similar_item(items, incoming_text, similarity_threshold, text_key="semantic_text")
            if matched is not None:
                matched["card_ref"] = card.to_dict()
                matched["title"] = str(card.title or matched.get("title", ""))
                matched["summary"] = _merge_text(str(matched.get("summary", "")), str(card.summary or ""))
                matched["tags"] = _merge_list(matched.get("tags", []), card.tags)
                matched["confidence"] = max(float(matched.get("confidence", 0.0) or 0.0), float(card.confidence or 0.0))
                matched["semantic_text"] = _merge_text(str(matched.get("semantic_text", "")), incoming_text)
                matched["updated_at"] = now_utc_iso()
                self._save_items(self.knowledge_path, items)
                return str(matched.get("knowledge_id", "")), False

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
                    "semantic_text": incoming_text,
                    "created_at": now_utc_iso(),
                    "updated_at": now_utc_iso(),
                }
            )
            self._save_items(self.knowledge_path, items)
            return entry_id, True

    def add_task(self, card: LiftedCard) -> str:
        task_id, _ = self.add_or_merge_task(card)
        return task_id

    def add_or_merge_task(self, card: LiftedCard, similarity_threshold: float = 0.72) -> tuple[str, bool]:
        with self._lock:
            items = self._load_items(self.task_path)
            incoming_text = _semantic_text(card)
            active_items = [item for item in items if str(item.get("status", "todo")) in {"todo", "in_progress", "blocked"}]
            matched = _find_most_similar_item(active_items, incoming_text, similarity_threshold, text_key="semantic_text")
            if matched is not None:
                matched["action"] = _merge_text(str(matched.get("action", "")), str(card.suggestion or ""))
                matched["owner_hint"] = _merge_list(matched.get("owner_hint", []), card.participants)
                matched["evidence"] = _merge_list(matched.get("evidence", []), card.evidence)[:10]
                matched["semantic_text"] = _merge_text(str(matched.get("semantic_text", "")), incoming_text)
                matched["updated_at"] = now_utc_iso()
                history = matched.get("history", [])
                if not isinstance(history, list):
                    history = []
                history.append({"status": str(matched.get("status", "todo")), "comment": "merged_similar_signal", "updated_at": matched["updated_at"]})
                matched["history"] = history[-50:]
                self._save_items(self.task_path, items)
                return str(matched.get("task_id", "")), False

            task_id = "task-" + hashlib.md5(card.card_id.encode("utf-8")).hexdigest()[:12]
            items.append(
                {
                    "card_id": card.card_id,
                    "task_id": task_id,
                    "action": card.suggestion,
                    "owner_hint": card.participants,
                    "deadline_hint": "",
                    "priority": _priority_from_confidence(card.confidence),
                    "evidence": card.evidence,
                    "semantic_text": incoming_text,
                    "status": "todo",
                    "created_at": now_utc_iso(),
                    "updated_at": now_utc_iso(),
                    "history": [],
                }
            )
            self._save_items(self.task_path, items)
            return task_id, True

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

    def merge_observe_items(self, source_observe_id: str, target_observe_id: str) -> dict[str, Any] | None:
        with self._lock:
            items = self._load_items(self.observe_path)
            source_id = str(source_observe_id or "").strip()
            target_id = str(target_observe_id or "").strip()
            if not source_id or not target_id or source_id == target_id:
                return None
            source = next((item for item in items if str(item.get("observe_id", "")) == source_id), None)
            target = next((item for item in items if str(item.get("observe_id", "")) == target_id), None)
            if source is None or target is None:
                return None
            evidence = target.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []
            source_evidence = source.get("evidence", [])
            if isinstance(source_evidence, list):
                for item in source_evidence:
                    text = str(item).strip()
                    if text and text not in evidence:
                        evidence.append(text)
            target["evidence"] = evidence[:10]
            target["hit_count"] = int(target.get("hit_count", 0)) + int(source.get("hit_count", 0))
            target["last_seen_at"] = now_utc_iso()
            source["pop_status"] = "popped"
            source["pop_count"] = int(source.get("pop_count", 0)) + 1
            source["last_popped_at"] = now_utc_iso()
            source["escalation_state"] = "merged_into_observe"
            self._save_items(self.observe_path, items)
            return target

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


_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+")


def _semantic_text(card: LiftedCard) -> str:
    participants = ", ".join([str(item).strip() for item in card.participants if str(item).strip()])
    return "\n".join(
        [
            f"title: {str(card.title or '').strip()}",
            f"topic_focus: {str(card.topic_focus or '').strip()}",
            f"summary: {str(card.summary or '').strip()}",
            f"times: {str(card.times or '').strip()}",
            f"locations: {str(card.locations or '').strip()}",
            f"participants: {participants}",
        ]
    ).strip()


def _tokens(text: str) -> set[str]:
    return {item.lower() for item in _WORD_RE.findall(str(text or "")) if len(item.strip()) >= 2}


def _similarity(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(max(1, len(a | b)))


def _find_most_similar_item(items: list[dict[str, Any]], incoming_text: str, threshold: float, text_key: str) -> dict[str, Any] | None:
    best_item: dict[str, Any] | None = None
    best_score = float(threshold)
    for item in items:
        current_text = str(item.get(text_key, "")).strip()
        if not current_text:
            continue
        score = _similarity(incoming_text, current_text)
        if score >= best_score:
            best_score = score
            best_item = item
    return best_item


def _merge_list(left: Any, right: list[str]) -> list[str]:
    out: list[str] = []
    if isinstance(left, list):
        out.extend([str(item).strip() for item in left if str(item).strip()])
    out.extend([str(item).strip() for item in (right or []) if str(item).strip()])
    uniq: list[str] = []
    for item in out:
        if item not in uniq:
            uniq.append(item)
    return uniq


def _merge_text(left: str, right: str) -> str:
    a = str(left or "").strip()
    b = str(right or "").strip()
    if not a:
        return b
    if not b:
        return a
    if b in a:
        return a
    return f"{a} | {b}"

__all__ = ["FeedbackSummary", "LocalStateStore"]
