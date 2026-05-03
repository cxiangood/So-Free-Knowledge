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
        append_sense: bool = False,
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
        existing_senses = self._normalize_senses(existing)
        matched_sense = self._find_matching_sense(
            existing_senses,
            entry_type=normalized_type,
            value=normalized_value if normalized_type != "nothing" else "",
        )

        if matched_sense is not None:
            matched_sense["aliases"] = _unique_strings(list(matched_sense.get("aliases", [])) + list(aliases or []))
            matched_sense["context_ids"] = _unique_strings(
                list(matched_sense.get("context_ids", [])) + list(context_ids or [])
            )
            if entity_id:
                matched_sense["entity_id"] = str(entity_id)
            matched_sense["source"] = str(source or matched_sense.get("source") or existing.get("source") or "manual")
            matched_sense["updated_at"] = now
        else:
            new_sense = {
                "sense_id": self._build_sense_id(
                    existing_senses=existing_senses,
                    entry_type=normalized_type,
                    value=normalized_value if normalized_type != "nothing" else "",
                ),
                "type": normalized_type,
                "value": normalized_value if normalized_type != "nothing" else "",
                "aliases": _unique_strings(aliases or []),
                "entity_id": str(entity_id or ""),
                "context_ids": _unique_strings(context_ids or []),
                "source": str(source or existing.get("source") or "manual"),
                "created_at": now,
                "updated_at": now,
            }
            if append_sense or not existing_senses:
                existing_senses.append(new_sense)
            else:
                existing_senses = [new_sense]

        primary_sense = existing_senses[-1] if existing_senses else {}
        merged = {
            "keyword": normalized_keyword,
            "type": str(primary_sense.get("type") or normalized_type),
            "value": str(primary_sense.get("value") or ""),
            "aliases": _unique_strings(
                [alias for sense in existing_senses for alias in sense.get("aliases", [])]
                + list(existing.get("aliases", []))
            ),
            "source": str(source or existing.get("source") or "manual"),
            "entity_id": str(primary_sense.get("entity_id") or entity_id or existing.get("entity_id") or ""),
            "context_ids": _unique_strings(
                [ctx_id for sense in existing_senses for ctx_id in sense.get("context_ids", [])]
                + list(existing.get("context_ids", []))
            ),
            "senses": existing_senses,
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
        senses = LingoStore._normalize_senses(value)
        primary_sense = senses[-1] if senses else {}
        return {
            "keyword": str(keyword),
            "type": str(value.get("type") or primary_sense.get("type") or "nothing"),
            "value": str(value.get("value") or primary_sense.get("value") or ""),
            "aliases": _unique_strings(value.get("aliases", []))
            or _unique_strings([alias for sense in senses for alias in sense.get("aliases", [])]),
            "source": str(value.get("source") or ""),
            "entity_id": str(value.get("entity_id") or primary_sense.get("entity_id") or ""),
            "context_ids": _unique_strings(value.get("context_ids", []))
            or _unique_strings([ctx_id for sense in senses for ctx_id in sense.get("context_ids", [])]),
            "senses": senses,
            "created_at": str(value.get("created_at") or ""),
            "updated_at": str(value.get("updated_at") or ""),
        }

    @staticmethod
    def _normalize_senses(value: dict[str, Any]) -> list[dict[str, Any]]:
        raw_senses = value.get("senses", [])
        senses: list[dict[str, Any]] = []
        if isinstance(raw_senses, list):
            for index, raw_sense in enumerate(raw_senses, start=1):
                if not isinstance(raw_sense, dict):
                    continue
                senses.append(
                    {
                        "sense_id": str(raw_sense.get("sense_id") or f"sense_{index}"),
                        "type": str(raw_sense.get("type") or "nothing"),
                        "value": str(raw_sense.get("value") or ""),
                        "aliases": _unique_strings(raw_sense.get("aliases", [])),
                        "entity_id": str(raw_sense.get("entity_id") or ""),
                        "context_ids": _unique_strings(raw_sense.get("context_ids", [])),
                        "source": str(raw_sense.get("source") or value.get("source") or ""),
                        "created_at": str(raw_sense.get("created_at") or value.get("created_at") or ""),
                        "updated_at": str(raw_sense.get("updated_at") or value.get("updated_at") or ""),
                    }
                )
        if senses:
            return senses
        entry_type = str(value.get("type") or "nothing")
        entry_value = str(value.get("value") or "")
        if not entry_value and entry_type == "nothing":
            return []
        return [
            {
                "sense_id": "sense_1",
                "type": entry_type,
                "value": entry_value,
                "aliases": _unique_strings(value.get("aliases", [])),
                "entity_id": str(value.get("entity_id") or ""),
                "context_ids": _unique_strings(value.get("context_ids", [])),
                "source": str(value.get("source") or ""),
                "created_at": str(value.get("created_at") or ""),
                "updated_at": str(value.get("updated_at") or ""),
            }
        ]

    @staticmethod
    def _find_matching_sense(
        senses: list[dict[str, Any]],
        *,
        entry_type: str,
        value: str,
    ) -> dict[str, Any] | None:
        for sense in senses:
            if str(sense.get("type") or "").strip().lower() == entry_type and str(sense.get("value") or "").strip() == value:
                return sense
        return None

    @staticmethod
    def _build_sense_id(
        *,
        existing_senses: list[dict[str, Any]],
        entry_type: str,
        value: str,
    ) -> str:
        base = f"{entry_type}:{value}".strip(":")
        safe = "".join(ch if ch.isalnum() else "_" for ch in base)[:40].strip("_") or "sense"
        existing_ids = {str(item.get("sense_id") or "") for item in existing_senses}
        if safe not in existing_ids:
            return safe
        suffix = 2
        while f"{safe}_{suffix}" in existing_ids:
            suffix += 1
        return f"{safe}_{suffix}"


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
