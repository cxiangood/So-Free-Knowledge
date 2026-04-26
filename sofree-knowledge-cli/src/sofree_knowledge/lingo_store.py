from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LingoStore:
    def __init__(self, output_dir: str | Path = ".") -> None:
        root = Path(output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "lingo_entries.json"

    def list_entries(self) -> list[dict[str, Any]]:
        data = self._load()
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            return []
        result = [self._normalize_entry(keyword, value) for keyword, value in entries.items()]
        result.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        return result

    def get_entry(self, keyword: str) -> dict[str, Any] | None:
        normalized = str(keyword or "").strip()
        if not normalized:
            return None
        data = self._load()
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            return None
        value = entries.get(normalized)
        if not isinstance(value, dict):
            return None
        return self._normalize_entry(normalized, value)

    def upsert_entry(
        self,
        keyword: str,
        entry_type: str,
        value: str,
        aliases: list[str] | None = None,
        source: str = "manual",
        entity_id: str = "",
        context_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_keyword = str(keyword or "").strip()
        if not normalized_keyword:
            raise ValueError("keyword is required")
        normalized_type = str(entry_type or "").strip().lower()
        if normalized_type not in {"key", "black", "confused", "nothing"}:
            raise ValueError("type must be one of: key, black, confused, nothing")
        normalized_value = str(value or "").strip()
        if normalized_type in {"key", "black"} and not normalized_value:
            raise ValueError("value is required for key/black entries")

        data = self._load()
        entries = data.setdefault("entries", {})
        if not isinstance(entries, dict):
            entries = {}
            data["entries"] = entries

        now = datetime.now(timezone.utc).isoformat()
        existing = entries.get(normalized_keyword, {})
        if not isinstance(existing, dict):
            existing = {}
        merged = {
            "keyword": normalized_keyword,
            "type": normalized_type,
            "value": normalized_value if normalized_type != "nothing" else "",
            "aliases": _unique_strings(aliases or existing.get("aliases", [])),
            "source": str(source or existing.get("source") or "manual"),
            "entity_id": str(entity_id or existing.get("entity_id") or ""),
            "context_ids": _unique_strings(context_ids or existing.get("context_ids", [])),
            "created_at": str(existing.get("created_at") or now),
            "updated_at": now,
        }
        entries[normalized_keyword] = merged
        self._save(data)
        return merged

    def delete_entry(self, keyword: str) -> dict[str, Any]:
        normalized_keyword = str(keyword or "").strip()
        if not normalized_keyword:
            raise ValueError("keyword is required")
        data = self._load()
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            return {"deleted": False, "keyword": normalized_keyword, "exists": False}
        existed = normalized_keyword in entries
        if existed:
            entries.pop(normalized_keyword, None)
            self._save(data)
        return {"deleted": bool(existed), "keyword": normalized_keyword, "exists": bool(existed)}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"entries": {}}
        try:
            data: Any = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return {"entries": {}}
        if not isinstance(data, dict):
            return {"entries": {}}
        data.setdefault("entries", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_entry(keyword: str, value: dict[str, Any]) -> dict[str, Any]:
        return {
            "keyword": str(keyword),
            "type": str(value.get("type") or "nothing"),
            "value": str(value.get("value") or ""),
            "aliases": _unique_strings(value.get("aliases", [])),
            "source": str(value.get("source") or ""),
            "entity_id": str(value.get("entity_id") or ""),
            "context_ids": _unique_strings(value.get("context_ids", [])),
            "created_at": str(value.get("created_at") or ""),
            "updated_at": str(value.get("updated_at") or ""),
        }


def _unique_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
