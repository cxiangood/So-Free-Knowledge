from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from utils import getenv

from ..msg.types import MessageEvent
from ..store.io import read_json, write_json
from .send import _load_feishu_client_class, _temporary_feishu_credentials


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _safe_name(value: Any) -> str:
    return str(value or "").strip()


def _safe_id(value: Any) -> str:
    return str(value or "").strip()


@dataclass(slots=True)
class IdentityMapConfig:
    state_dir: str | Path
    env_file: str = ""
    receive_id_type: str = "user_id"
    bootstrap_on_empty: bool = True
    bootstrap_max_chats: int = 200


class UserIdentityMap:
    def __init__(self, config: IdentityMapConfig) -> None:
        self.config = config
        self.path = Path(config.state_dir) / "user_identity_map.json"
        self._lock = RLock()
        self._bootstrapped = False

    def ensure_bootstrap(self, *, app_id: str, app_secret: str) -> None:
        if self._bootstrapped or not self.config.bootstrap_on_empty:
            return
        with self._lock:
            if self._bootstrapped:
                return
            data = self._load()
            if self._is_valid(data):
                self._bootstrapped = True
                return
            self._bootstrap_from_feishu(app_id=app_id, app_secret=app_secret)
            self._bootstrapped = True

    def update_from_event(self, event: MessageEvent) -> None:
        chat_id = _safe_id(event.chat_id)
        sender_name = _safe_name(event.sender_name)
        sender_ids = {
            "user_id": _safe_id(event.sender_user_id),
            "open_id": _safe_id(event.sender_open_id),
            "union_id": _safe_id(event.sender_union_id),
        }
        with self._lock:
            data = self._load()
            self._touch_identity(data, name=sender_name, ids=sender_ids, chat_id=chat_id)
            for item in event.mentions:
                if not isinstance(item, dict):
                    continue
                mention_ids = item.get("id", {})
                if not isinstance(mention_ids, dict):
                    mention_ids = {}
                self._touch_identity(
                    data,
                    name=_safe_name(item.get("name", "")),
                    ids={
                        "user_id": _safe_id(mention_ids.get("user_id", "")),
                        "open_id": _safe_id(mention_ids.get("open_id", "")),
                        "union_id": _safe_id(mention_ids.get("union_id", "")),
                    },
                    chat_id=chat_id,
                )
            self._save(data)

    def resolve_name_in_chat(self, *, chat_id: str, name: str) -> tuple[str, str]:
        norm = _normalize_name(name)
        if not norm:
            return "", "empty_name"
        with self._lock:
            data = self._load()
            chat_map = data.get("name_in_chat_index", {})
            if not isinstance(chat_map, dict):
                return "", "chat_index_missing"
            chat_entry = chat_map.get(str(chat_id), {})
            if not isinstance(chat_entry, dict):
                return "", "chat_not_found"
            ids = chat_entry.get(norm, [])
            ids = [str(v).strip() for v in ids if str(v).strip()]
            uniq = sorted(set(ids))
            if len(uniq) == 1:
                return uniq[0], ""
            if len(uniq) > 1:
                return "", "ambiguous_name_in_chat"
            return "", "name_not_found_in_chat"

    def _bootstrap_from_feishu(self, *, app_id: str, app_secret: str) -> None:
        FeishuClient = _load_feishu_client_class()
        data = self._empty()
        try:
            with _temporary_feishu_credentials(app_id, app_secret):
                client = FeishuClient()
                chats = self._list_all_chats(client)
                for chat in chats[: max(1, int(self.config.bootstrap_max_chats))]:
                    chat_id = _safe_id(chat.get("chat_id", ""))
                    if not chat_id:
                        continue
                    members = self._list_chat_members(client, chat_id=chat_id)
                    for member in members:
                        member_id = member.get("member_id", {})
                        if not isinstance(member_id, dict):
                            member_id = {}
                        self._touch_identity(
                            data,
                            name=_safe_name(member.get("name", "")),
                            ids={
                                "user_id": _safe_id(member_id.get("user_id", "")),
                                "open_id": _safe_id(member_id.get("open_id", "")),
                                "union_id": _safe_id(member_id.get("union_id", "")),
                            },
                            chat_id=chat_id,
                        )
        except Exception:
            # Bootstrap is best-effort; keep empty file so runtime can grow incrementally.
            pass
        self._save(data)

    def _list_all_chats(self, client: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page_token = ""
        while True:
            payload = client.request(
                "GET",
                "/open-apis/im/v1/chats",
                access_token=client.get_tenant_access_token(),
                params={"user_id_type": "open_id", "page_size": 100, **({"page_token": page_token} if page_token else {})},
            )
            data = payload.get("data", payload)
            items = data.get("items", [])
            if isinstance(items, list):
                rows.extend([item for item in items if isinstance(item, dict)])
            has_more = bool(data.get("has_more", False))
            page_token = str(data.get("page_token", "") or "")
            if not has_more or not page_token:
                break
        return rows

    def _list_chat_members(self, client: Any, *, chat_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page_token = ""
        while True:
            payload = client.request(
                "GET",
                f"/open-apis/im/v1/chats/{chat_id}/members",
                access_token=client.get_tenant_access_token(),
                params={
                    "member_id_type": "open_id",
                    "user_id_type": "user_id",
                    "page_size": 50,
                    **({"page_token": page_token} if page_token else {}),
                },
            )
            data = payload.get("data", payload)
            items = data.get("items", [])
            if isinstance(items, list):
                rows.extend([item for item in items if isinstance(item, dict)])
            has_more = bool(data.get("has_more", False))
            page_token = str(data.get("page_token", "") or "")
            if not has_more or not page_token:
                break
        return rows

    def _touch_identity(self, data: dict[str, Any], *, name: str, ids: dict[str, str], chat_id: str) -> None:
        canonical = _safe_id(ids.get(self.config.receive_id_type, "")) or _safe_id(ids.get("user_id", "")) or _safe_id(ids.get("open_id", ""))
        if not canonical:
            return
        id_index = data.setdefault("id_index", {})
        aliases_by_id = data.setdefault("aliases_by_id", {})
        name_global = data.setdefault("name_global_index", {})
        name_in_chat = data.setdefault("name_in_chat_index", {})
        profile = id_index.get(canonical)
        if not isinstance(profile, dict):
            profile = {}
        profile["canonical_id"] = canonical
        profile["receive_id_type"] = self.config.receive_id_type
        profile["user_id"] = _safe_id(ids.get("user_id", "")) or profile.get("user_id", "")
        profile["open_id"] = _safe_id(ids.get("open_id", "")) or profile.get("open_id", "")
        profile["union_id"] = _safe_id(ids.get("union_id", "")) or profile.get("union_id", "")
        if name:
            profile["display_name"] = name
        id_index[canonical] = profile

        alias_set = set(str(v).strip() for v in aliases_by_id.get(canonical, []) if str(v).strip())
        if name:
            alias_set.add(name)
        aliases_by_id[canonical] = sorted(alias_set)

        for alias in aliases_by_id[canonical]:
            norm = _normalize_name(alias)
            if not norm:
                continue
            g_ids = set(str(v).strip() for v in name_global.get(norm, []) if str(v).strip())
            g_ids.add(canonical)
            name_global[norm] = sorted(g_ids)

            chat_norm = str(chat_id or "").strip()
            if chat_norm:
                c_map = name_in_chat.get(chat_norm)
                if not isinstance(c_map, dict):
                    c_map = {}
                c_ids = set(str(v).strip() for v in c_map.get(norm, []) if str(v).strip())
                c_ids.add(canonical)
                c_map[norm] = sorted(c_ids)
                name_in_chat[chat_norm] = c_map

    def _is_valid(self, data: dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        return isinstance(data.get("id_index", {}), dict) and isinstance(data.get("name_in_chat_index", {}), dict)

    def _empty(self) -> dict[str, Any]:
        return {
            "receive_id_type": self.config.receive_id_type,
            "id_index": {},
            "name_global_index": {},
            "name_in_chat_index": {},
            "aliases_by_id": {},
        }

    def _load(self) -> dict[str, Any]:
        data = read_json(self.path, self._empty())
        if not self._is_valid(data):
            return self._empty()
        return data

    def _save(self, data: dict[str, Any]) -> None:
        write_json(self.path, data)


def parse_participants_names(raw: Any) -> list[str]:
    if isinstance(raw, list):
        rows: list[str] = []
        seen: set[str] = set()
        for item in raw:
            name = str(item or "").strip()
            if not name:
                continue
            key = _normalize_name(name)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(name)
        return rows
    text = str(raw or "").strip()
    if not text:
        return []
    parsed = _parse_python_list_names(text)
    if parsed:
        return parsed
    parts = re.split(r"[，,、;；/|\n\r\t ]+", text)
    rows: list[str] = []
    seen: set[str] = set()
    for part in parts:
        name = part.strip()
        if not name:
            continue
        key = _normalize_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(name)
    return rows


def _parse_python_list_names(text: str) -> list[str]:
    candidate = str(text or "").strip()
    if not (candidate.startswith("[") and candidate.endswith("]")):
        return []
    try:
        value = ast.literal_eval(candidate)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    seen: set[str] = set()
    for item in value:
        name = str(item or "").strip()
        if not name:
            continue
        key = _normalize_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(name)
    return rows


def resolve_identity_map_config(*, state_dir: str | Path, env_file: str = "") -> IdentityMapConfig:
    from .feishu import load_env_file

    load_env_file(env_file)
    return IdentityMapConfig(
        state_dir=state_dir,
        env_file=env_file,
        receive_id_type=(getenv("TASK_PUSH_RECEIVE_ID_TYPE", "user_id") or "user_id").strip(),
        bootstrap_on_empty=(getenv("TASK_PUSH_BOOTSTRAP_ON_EMPTY", "true").strip().lower() not in {"0", "false", "no", "off"}),
        bootstrap_max_chats=max(1, int(getenv("TASK_PUSH_BOOTSTRAP_MAX_CHATS", "200") or 200)),
    )


__all__ = [
    "IdentityMapConfig",
    "UserIdentityMap",
    "parse_participants_names",
    "resolve_identity_map_config",
]
