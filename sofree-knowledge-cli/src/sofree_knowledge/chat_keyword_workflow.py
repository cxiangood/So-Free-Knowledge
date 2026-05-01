from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from .archive import collect_messages
from .config import load_env_file, resolve_env_file
from .feishu_client import FeishuClient
from . import wikisheet as wikisheet_module


HEADER_COLUMNS = [
    "run_id",
    "chat_id",
    "sheet_title",
    "keyword",
    "type",
    "sense",
    "count",
    "ratio",
    "contexts_json",
    "source_message_count",
    "created_at_utc",
]


def _ensure_repo_root_on_sys_path() -> Path:
    # .../sofree-knowledge-cli/src/sofree_knowledge/chat_keyword_workflow.py -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return repo_root


def _load_message_extract_functions() -> tuple[Callable[[Path], list[dict[str, Any]]], Callable[..., list[str]]]:
    try:
        from message_extract.extract_chat_messages import load_records, extract_plain_messages

        return load_records, extract_plain_messages
    except Exception:
        _ensure_repo_root_on_sys_path()
        try:
            from message_extract.extract_chat_messages import load_records, extract_plain_messages

            return load_records, extract_plain_messages
        except Exception as exc:  # pragma: no cover - guarded by actionable error text
            raise RuntimeError(
                "Failed to import message_extract.extract_chat_messages. "
                "Ensure repository root is available and module path is correct."
            ) from exc


def _load_classify_function() -> Callable[[Any, dict[str, Any] | None], dict[str, Any]]:
    try:
        from token_classify.classify import classify

        return classify
    except Exception:
        _ensure_repo_root_on_sys_path()
        try:
            from token_classify.classify import classify

            return classify
        except Exception as exc:  # pragma: no cover - guarded by actionable error text
            raise RuntimeError(
                "Failed to import token_classify.classify.classify. "
                "Ensure repository root is available and module path is correct."
            ) from exc


def _normalize_sheet_title(chat_id: str, sheet_title: str) -> str:
    normalized = str(sheet_title or "").strip()
    if normalized:
        return normalized
    return f"chat_keywords_{str(chat_id).strip()}"


def _is_sheet_item(item: dict[str, Any]) -> bool:
    file_type = str(item.get("type") or item.get("file_type") or item.get("obj_type") or "").strip().lower()
    if file_type:
        return file_type == "sheet"
    url = str(item.get("url") or "").lower()
    return "/sheets/" in url


def _extract_sheet_token(item: dict[str, Any]) -> str:
    return str(
        item.get("token")
        or item.get("file_token")
        or item.get("obj_token")
        or item.get("spreadsheet_token")
        or ""
    ).strip()


def _find_sheet_by_title(client: FeishuClient, sheet_title: str, max_scan: int = 500) -> dict[str, Any] | None:
    scanned = 0
    page_token = ""
    while scanned < max_scan:
        page = client.list_drive_files(page_size=min(50, max_scan - scanned), page_token=page_token)
        items = page.get("items", [])
        if not isinstance(items, list):
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            scanned += 1
            title = str(item.get("name") or item.get("title") or "").strip()
            if title != sheet_title:
                continue
            if not _is_sheet_item(item):
                continue
            token = _extract_sheet_token(item)
            if not token:
                continue
            return {"title": title, "spreadsheet_token": token, "url": str(item.get("url") or ""), "raw": item}
        page_token = str(page.get("page_token") or "")
        if not page.get("has_more") or not page_token:
            break
    return None


def _first_sheet_id_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        if isinstance(payload.get("sheets"), list):
            sheets = payload["sheets"]
            if sheets and isinstance(sheets[0], dict):
                first = sheets[0]
                return str(first.get("sheet_id") or first.get("sheetId") or first.get("id") or "").strip()
        for value in payload.values():
            sheet_id = _first_sheet_id_from_payload(value)
            if sheet_id:
                return sheet_id
    elif isinstance(payload, list):
        for item in payload:
            sheet_id = _first_sheet_id_from_payload(item)
            if sheet_id:
                return sheet_id
    return ""


def _get_first_sheet_id(client: FeishuClient, spreadsheet_token: str) -> str:
    token = str(spreadsheet_token or "").strip()
    if not token:
        return ""
    attempts = [
        ("GET", f"/open-apis/sheets/v2/spreadsheets/{token}/metainfo", None),
        ("GET", f"/open-apis/sheets/v2/spreadsheets/{token}", None),
        ("GET", f"/open-apis/sheets/v3/spreadsheets/{token}", None),
    ]
    for method, path, params in attempts:
        try:
            data = client.request(method, path, access_token=client.get_tenant_access_token(), params=params)
        except Exception:
            continue
        sheet_id = _first_sheet_id_from_payload(data.get("data", data))
        if sheet_id:
            return sheet_id
    return ""


def _extract_values_from_payload(payload: Any) -> list[list[Any]]:
    if not isinstance(payload, dict):
        return []
    value_range = payload.get("valueRange")
    if isinstance(value_range, dict) and isinstance(value_range.get("values"), list):
        rows = value_range.get("values")
        return rows if isinstance(rows, list) else []
    if isinstance(payload.get("values"), list):
        rows = payload.get("values")
        return rows if isinstance(rows, list) else []
    value_ranges = payload.get("valueRanges")
    if isinstance(value_ranges, list) and value_ranges and isinstance(value_ranges[0], dict):
        first = value_ranges[0]
        if isinstance(first.get("values"), list):
            rows = first.get("values")
            return rows if isinstance(rows, list) else []
    return []


def _best_effort_read_first_cell(client: FeishuClient, spreadsheet_token: str, read_range: str) -> str | None:
    token = str(spreadsheet_token or "").strip()
    normalized_range = str(read_range or "").strip()
    if not token or not normalized_range:
        return None
    encoded_range = quote(normalized_range, safe="")
    attempts: list[tuple[str, str, dict[str, Any] | None]] = [
        ("GET", f"/open-apis/sheets/v2/spreadsheets/{token}/values/{encoded_range}", None),
        ("GET", f"/open-apis/sheets/v2/spreadsheets/{token}/values", {"range": normalized_range}),
        ("GET", f"/open-apis/sheets/v2/spreadsheets/{token}/values_batch_get", {"ranges": normalized_range}),
    ]
    for method, path, params in attempts:
        try:
            data = client.request(method, path, access_token=client.get_tenant_access_token(), params=params)
        except Exception:
            continue
        rows = _extract_values_from_payload(data.get("data", data))
        if not rows:
            return ""
        first_row = rows[0] if rows and isinstance(rows[0], list) else []
        if not first_row:
            return ""
        return str(first_row[0] if first_row[0] is not None else "")
    return None


def _sheet_appears_empty(client: FeishuClient, spreadsheet_token: str, sheet_id: str) -> bool:
    read_range = f"{sheet_id}!A1:A1" if sheet_id else "A1:A1"
    first_cell = _best_effort_read_first_cell(client, spreadsheet_token, read_range)
    if first_cell is None:
        # Cannot determine safely; avoid writing duplicate header.
        return False
    return not str(first_cell).strip()


def _flatten_classification_results(
    *,
    run_id: str,
    chat_id: str,
    sheet_title: str,
    classification_results: dict[str, Any],
    source_message_count: int,
    created_at_utc: str,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for keyword, entries in classification_results.items():
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            entry_type = str(item.get("type") or "")
            sense = str(item.get("sense") or "")
            count_raw = item.get("count", 0)
            ratio_raw = item.get("ratio", 0)
            contexts = item.get("contexts", [])
            count: int
            ratio: float
            try:
                count = int(count_raw)
            except (TypeError, ValueError):
                count = 0
            try:
                ratio = float(ratio_raw)
            except (TypeError, ValueError):
                ratio = 0.0
            contexts_json = json.dumps(contexts if isinstance(contexts, list) else [], ensure_ascii=False)
            rows.append(
                [
                    run_id,
                    chat_id,
                    sheet_title,
                    str(keyword),
                    entry_type,
                    sense,
                    count,
                    ratio,
                    contexts_json,
                    source_message_count,
                    created_at_utc,
                ]
            )
    return rows


def _make_wikisheet_args(env_file: str, output_dir: str, **kwargs: Any) -> argparse.Namespace:
    payload = {"env_file": env_file, "output_dir": output_dir}
    payload.update(kwargs)
    return argparse.Namespace(**payload)


def run_chat_keyword_to_wikisheet_workflow(
    *,
    chat_id: str,
    env_file: str = "",
    output_dir: str = ".",
    start_time: str = "",
    end_time: str = "",
    max_messages_per_chat: int = 1000,
    page_size: int = 50,
    top_keywords: int = 20,
    text_window_tokens: int = 80,
    message_window_sentences: int = 4,
    sheet_title: str = "",
    space_id: str = "",
    parent_node_token: str = "",
) -> dict[str, Any]:
    normalized_chat_id = str(chat_id or "").strip()
    if not normalized_chat_id:
        raise ValueError("chat_id is required")

    env_path = resolve_env_file(env_file, output_dir=output_dir)
    load_env_file(env_path)

    client = FeishuClient()
    manifest = collect_messages(
        client=client,
        output_dir=output_dir,
        output_subdir="",
        chat_ids=normalized_chat_id,
        include_visible_chats=False,
        start_time=start_time,
        end_time=end_time,
        max_chats=1,
        max_messages_per_chat=max_messages_per_chat,
        page_size=page_size,
    )

    messages_path = str(((manifest.get("files") or {}) if isinstance(manifest, dict) else {}).get("messages") or "")
    if not messages_path:
        archive_dir = str(manifest.get("archive_dir") or "").strip() if isinstance(manifest, dict) else ""
        if archive_dir:
            messages_path = str(Path(archive_dir) / "messages.jsonl")
    if not messages_path:
        raise RuntimeError("collect_messages returned no messages file path")

    load_records, extract_plain_messages = _load_message_extract_functions()
    records = load_records(Path(messages_path))
    plain_messages = extract_plain_messages(records, include_types={"text", "post"})
    source_message_count = len(plain_messages)

    classification_results: dict[str, Any] = {}
    classification_top_keywords: list[str] = []
    if plain_messages:
        classify = _load_classify_function()
        classify_result = classify(
            plain_messages,
            {
                "top_keywords": top_keywords,
                "text_window_tokens": text_window_tokens,
                "message_window_sentences": message_window_sentences,
            },
        )
        if isinstance(classify_result, dict):
            loaded_results = classify_result.get("classification_results", {})
            if isinstance(loaded_results, dict):
                classification_results = loaded_results
            top_list = classify_result.get("top_keywords", [])
            if isinstance(top_list, list):
                classification_top_keywords = [str(item) for item in top_list]

    normalized_title = _normalize_sheet_title(normalized_chat_id, sheet_title)
    existing = _find_sheet_by_title(client, normalized_title)
    created_or_reused = "reused"
    spreadsheet_token = ""
    spreadsheet_url = ""
    if existing:
        spreadsheet_token = str(existing.get("spreadsheet_token") or "")
        spreadsheet_url = str(existing.get("url") or "")
    else:
        created_or_reused = "created"
        create_args = _make_wikisheet_args(
            env_file=env_file,
            output_dir=output_dir,
            title=normalized_title,
            space_id=space_id,
            parent_node_token=parent_node_token,
        )
        created = wikisheet_module.cmd_create_sheet(create_args)
        spreadsheet_token = str(created.get("spreadsheet_token") or "")
        spreadsheet_url = str(created.get("url") or "")
    if not spreadsheet_token:
        raise RuntimeError("Failed to resolve spreadsheet_token")

    run_id = str(manifest.get("run_id") or "").strip() if isinstance(manifest, dict) else ""
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    created_at_utc = datetime.now(timezone.utc).isoformat()
    rows = _flatten_classification_results(
        run_id=run_id,
        chat_id=normalized_chat_id,
        sheet_title=normalized_title,
        classification_results=classification_results,
        source_message_count=source_message_count,
        created_at_utc=created_at_utc,
    )

    first_sheet_id = _get_first_sheet_id(client, spreadsheet_token)
    append_anchor = f"{first_sheet_id}!A1" if first_sheet_id else "A1"
    header_appended = False
    if _sheet_appears_empty(client, spreadsheet_token, first_sheet_id):
        header_args = _make_wikisheet_args(
            env_file=env_file,
            output_dir=output_dir,
            spreadsheet_token=spreadsheet_token,
            range=append_anchor,
            values=json.dumps([HEADER_COLUMNS], ensure_ascii=False),
        )
        wikisheet_module.cmd_append_data(header_args)
        header_appended = True

    if rows:
        row_args = _make_wikisheet_args(
            env_file=env_file,
            output_dir=output_dir,
            spreadsheet_token=spreadsheet_token,
            range=append_anchor,
            values=json.dumps(rows, ensure_ascii=False),
        )
        wikisheet_module.cmd_append_data(row_args)

    if not classification_top_keywords and classification_results:
        classification_top_keywords = list(classification_results.keys())

    return {
        "ok": True,
        "chat_id": normalized_chat_id,
        "run_id": run_id,
        "archive_dir": str(manifest.get("archive_dir") or "") if isinstance(manifest, dict) else "",
        "messages_file": messages_path,
        "plain_message_count": source_message_count,
        "classification_keyword_count": len(classification_results),
        "classification_top_keywords": classification_top_keywords,
        "rows_written": len(rows),
        "header_appended": header_appended,
        "sheet_title": normalized_title,
        "spreadsheet_token": spreadsheet_token,
        "spreadsheet_url": spreadsheet_url,
        "created_or_reused": created_or_reused,
        "first_sheet_id": first_sheet_id,
        "append_anchor": append_anchor,
    }

