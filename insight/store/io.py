from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

_PATH_LOCKS: dict[str, RLock] = {}
_PATH_LOCKS_GUARD = RLock()


def _lock_for(path: Path) -> RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _PATH_LOCKS[key] = lock
    return lock


def write_json(path: Path, payload: Any) -> None:
    lock = _lock_for(path)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lock = _lock_for(path)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lock = _lock_for(path)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any) -> Any:
    lock = _lock_for(path)
    with lock:
        if not path.exists():
            return default
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return default
        return data


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    lock = _lock_for(path)
    with lock:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                obj = json.loads(text)
                if isinstance(obj, dict):
                    rows.append(obj)
            return rows
    data = read_json(path, default=[])
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


__all__ = ["write_json", "write_jsonl", "append_jsonl", "read_json", "read_json_or_jsonl"]
