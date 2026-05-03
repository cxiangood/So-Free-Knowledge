from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .archive import collect_messages
from .lingo_context import extract_keyword_contexts, publishable_lingo_judgements
from .lingo_store import LingoStore

DEFAULT_RECENT_DAYS = 7
DEFAULT_MIN_RUN_INTERVAL_DAYS = 7
DEFAULT_TOP_KEYWORDS = 30
DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_MIN_CONTEXTS = 1
DEFAULT_CONTEXT_BEFORE = 1
DEFAULT_CONTEXT_AFTER = 1
DEFAULT_MAX_CONTEXTS = 80
AUTO_JUDGEMENT_DECISIONS = {"create_entry", "append_new_sense", "skip_duplicate", "skip_noise", "skip_uncertain"}


class LingoAutoStateStore:
    def __init__(self, output_dir: str | Path = ".") -> None:
        root = Path(output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "lingo_auto_state.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def save(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_lingo_auto_pipeline(
    *,
    client: Any,
    output_dir: str | Path,
    recent_days: int = DEFAULT_RECENT_DAYS,
    min_run_interval_days: int = DEFAULT_MIN_RUN_INTERVAL_DAYS,
    force: bool = False,
    chat_ids: str | list[str] | None = None,
    include_visible_chats: bool = True,
    start_time: str = "",
    end_time: str = "",
    max_chats: int = 200,
    max_messages_per_chat: int = 500,
    page_size: int = 50,
    top_keywords: int = DEFAULT_TOP_KEYWORDS,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    min_contexts: int = DEFAULT_MIN_CONTEXTS,
    context_before: int = DEFAULT_CONTEXT_BEFORE,
    context_after: int = DEFAULT_CONTEXT_AFTER,
    max_contexts: int = DEFAULT_MAX_CONTEXTS,
    classifier_enabled: bool = True,
    analyzer_enabled: bool = True,
    classifier_config: dict[str, Any] | None = None,
    analyzer_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser()
    state_store = LingoAutoStateStore(output_root)
    state = state_store.load()
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    last_success_at = _parse_datetime(state.get("last_success_at"))
    next_allowed_at = None
    if last_success_at is not None:
        next_allowed_at = last_success_at + timedelta(days=max(0, int(min_run_interval_days)))
    if not force and next_allowed_at is not None and now < next_allowed_at:
        return {
            "ok": True,
            "skipped": True,
            "skip_reason": "min_run_interval_not_reached",
            "last_success_at": last_success_at.isoformat(),
            "next_allowed_at": next_allowed_at.isoformat(),
            "state_file": str(state_store.path),
        }

    window_start, window_end = _resolve_time_window(
        recent_days=recent_days,
        start_time=start_time,
        end_time=end_time,
        now=now,
    )
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    archive_manifest = collect_messages(
        client=client,
        output_dir=output_root,
        output_subdir=f"lingo_auto_{run_id}",
        chat_ids=chat_ids,
        include_visible_chats=include_visible_chats,
        start_time=window_start,
        end_time=window_end,
        max_chats=max_chats,
        max_messages_per_chat=max_messages_per_chat,
        page_size=page_size,
    )
    archive_dir = Path(str(archive_manifest.get("archive_dir") or "")).expanduser()
    messages = _load_archive_messages(archive_dir)
    classifier_result = _classify_messages(
        messages,
        config={
            "top_keywords": top_keywords,
            "enable_analyzer": analyzer_enabled,
            "analyzer_config": dict(analyzer_config or {}),
            "classifier_config": {
                "enabled": classifier_enabled,
                **dict(classifier_config or {}),
            },
        },
    )
    candidates = build_lingo_candidates(
        messages=messages,
        classifier_result=classifier_result,
        output_dir=output_root,
        candidate_limit=candidate_limit,
        min_frequency=min_frequency,
        min_contexts=min_contexts,
        context_before=context_before,
        context_after=context_after,
        max_contexts=max_contexts,
    )
    prompt = build_lingo_openclaw_prompt(candidates)

    run_dir = output_root / "lingo_auto_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "run_dir": str(run_dir),
        "candidates_file": str(run_dir / "candidates.json"),
        "classifier_result_file": str(run_dir / "classifier_result.json"),
        "openclaw_prompt_file": str(run_dir / "openclaw_prompt.txt"),
    }
    (run_dir / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "classifier_result.json").write_text(
        json.dumps(classifier_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "openclaw_prompt.txt").write_text(prompt, encoding="utf-8")

    state_store.save(
        {
            "last_run_at": now_iso,
            "last_success_at": now_iso,
            "last_run_id": run_id,
            "last_window_start": window_start,
            "last_window_end": window_end,
            "last_archive_dir": str(archive_dir),
            "last_run_dir": str(run_dir),
            "last_candidate_count": len(candidates),
        }
    )
    return {
        "ok": True,
        "skipped": False,
        "run_id": run_id,
        "window": {
            "recent_days": int(recent_days),
            "start_time": window_start,
            "end_time": window_end,
        },
        "archive": archive_manifest,
        "message_count": len(messages),
        "classifier_statistics": classifier_result.get("statistics", {}),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "openclaw": {
            "task": "review_lingo_candidates_and_decide_create_or_append_sense",
            "prompt": prompt,
            "expected_output_schema": {
                "type": "array",
                "item_fields": [
                    "keyword",
                    "decision",
                    "type",
                    "value",
                    "context_ids",
                    "matched_existing_sense_ids",
                    "aliases",
                    "reason",
                ],
            },
        },
        "artifacts": artifacts,
        "state_file": str(state_store.path),
    }


def build_lingo_candidates(
    *,
    messages: list[dict[str, Any]],
    classifier_result: dict[str, Any],
    output_dir: str | Path,
    candidate_limit: int,
    min_frequency: int,
    min_contexts: int,
    context_before: int,
    context_after: int,
    max_contexts: int,
) -> list[dict[str, Any]]:
    store = LingoStore(output_dir)
    keywords = [
        str(item).strip()
        for item in classifier_result.get("top_keywords", [])
        if str(item).strip()
    ]
    if not keywords:
        return []

    contexts = extract_keyword_contexts(
        keywords=keywords,
        messages=messages,
        before=context_before,
        after=context_after,
        max_contexts=max_contexts,
    )
    contexts_by_keyword: dict[str, list[dict[str, Any]]] = {keyword: [] for keyword in keywords}
    for context in contexts:
        for keyword in context.get("keywords", []):
            normalized = str(keyword).strip()
            if normalized in contexts_by_keyword:
                contexts_by_keyword[normalized].append(context)

    word_frequency = {
        str(keyword): int(count)
        for keyword, count in classifier_result.get("word_frequency", [])
        if str(keyword).strip()
    }
    token_scores = classifier_result.get("semantic_filter_details", {}).get("token_scores", {})
    sense_results = classifier_result.get("classification_results", {})

    candidates: list[dict[str, Any]] = []
    for keyword in keywords:
        if not _is_candidate_keyword(keyword):
            continue
        frequency = int(word_frequency.get(keyword, 0))
        if frequency < int(min_frequency):
            continue
        keyword_contexts = contexts_by_keyword.get(keyword, [])
        if len(keyword_contexts) < int(min_contexts):
            continue
        top_sense = {}
        senses = sense_results.get(keyword, [])
        if isinstance(senses, list) and senses:
            top_sense = senses[0] if isinstance(senses[0], dict) else {}
        score_payload = token_scores.get(keyword, {}) if isinstance(token_scores, dict) else {}
        candidate = {
            "keyword": keyword,
            "frequency": frequency,
            "context_count": len(keyword_contexts),
            "context_ids": [str(item.get("context_id") or "") for item in keyword_contexts if str(item.get("context_id") or "").strip()],
            "contexts": [
                {
                    "context_id": str(item.get("context_id") or ""),
                    "text": str(item.get("text") or ""),
                    "message_ids": [str(msg_id) for msg_id in item.get("message_ids", []) if str(msg_id).strip()],
                }
                for item in keyword_contexts
            ],
            "semantic_density": _coerce_float(score_payload.get("semantic_density")),
            "attention_entropy": _coerce_float(score_payload.get("attention_entropy")),
            "bert_score": _coerce_float(score_payload.get("score")),
            "initial_type": str(top_sense.get("type") or "confused"),
            "initial_value": str(top_sense.get("sense") or "").strip(),
            "initial_ratio": _coerce_float(top_sense.get("ratio")),
            "existing_entry": store.get_entry(keyword),
            "related_existing_entries": _find_related_existing_entries(store, keyword),
            "source": "lingo_auto_pipeline",
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -int(item.get("frequency", 0)),
            -int(item.get("context_count", 0)),
            -_coerce_float(item.get("bert_score")),
            str(item.get("keyword") or ""),
        )
    )
    return candidates[: max(0, int(candidate_limit))]


def build_lingo_openclaw_prompt(candidates: list[dict[str, Any]]) -> str:
    payload = {
        "task": "review_lingo_candidates_and_decide_create_or_append_sense",
        "candidates": [
            {
                "keyword": item.get("keyword", ""),
                "frequency": item.get("frequency", 0),
                "context_count": item.get("context_count", 0),
                "context_ids": item.get("context_ids", []),
                "semantic_density": item.get("semantic_density", 0.0),
                "attention_entropy": item.get("attention_entropy", 0.0),
                "bert_score": item.get("bert_score", 0.0),
                "initial_type": item.get("initial_type", "confused"),
                "initial_value": item.get("initial_value", ""),
                "initial_ratio": item.get("initial_ratio", 0.0),
                "existing_entry": _compact_entry(item.get("existing_entry")),
                "related_existing_entries": [
                    _compact_entry(existing)
                    for existing in item.get("related_existing_entries", [])
                    if isinstance(existing, dict)
                ],
                "contexts": [
                    {
                        "context_id": ctx.get("context_id", ""),
                        "text": ctx.get("text", ""),
                    }
                    for ctx in item.get("contexts", [])[:4]
                    if isinstance(ctx, dict)
                ],
            }
            for item in candidates
        ],
    }
    return (
        "你是 SoFree 的飞书词典审稿助手。目标是审查候选词条，并决定：\n"
        "1. 应该新建词条。\n"
        "2. 应该给已有词条追加一个新释义。\n"
        "3. 其实和已有释义重复，应跳过。\n"
        "4. 本身是噪音词或证据不足，应跳过。\n\n"
        "你必须重点检查重复和近似重复：\n"
        "- 若候选词的释义与已有释义表达接近、只是措辞不同，判为 skip_duplicate。\n"
        "- 若同一个 keyword 已有词条，但上下文明显体现出另一种含义，判为 append_new_sense。\n"
        "- 若没有合适的已有释义，且该词确实值得沉淀，判为 create_entry。\n\n"
        "只输出 JSON 数组，不要输出任何额外说明。\n"
        "每个元素必须包含字段：\n"
        "- keyword: 候选词\n"
        "- decision: create_entry / append_new_sense / skip_duplicate / skip_noise / skip_uncertain\n"
        "- type: key / black / confused / nothing\n"
        "- value: 适合飞书词典直接写入的简洁中文释义；如果跳过则为空字符串\n"
        "- context_ids: 支持该结论的 context_id 列表\n"
        "- matched_existing_sense_ids: 如果和已有释义相关，填命中的 sense_id 列表；否则空数组\n"
        "- aliases: 明确出现的别名列表，没有则空数组\n"
        "- reason: 不超过 40 字，解释为什么这样判\n\n"
        "判定规则：\n"
        "- create_entry: 新概念或新黑话，需要新建词条\n"
        "- append_new_sense: 同词多义，需要在已有词条下新增释义\n"
        "- skip_duplicate: 与已有释义实质重复，不新增\n"
        "- skip_noise: 噪音词、普通词、上下文价值不足\n"
        "- skip_uncertain: 有价值但证据仍不足，暂不写入\n"
        "- type 只能是 key / black / confused / nothing\n"
        "- decision 为 skip_* 时，value 必须为空字符串\n"
        "- decision 为 append_new_sense 或 create_entry 时，value 必须是可直接入库的定义\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_lingo_openclaw_judgements(raw: str | list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    data: Any = raw
    if isinstance(raw, str):
        data = json.loads(_strip_code_fence(raw))
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            data = data["items"]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("openclaw lingo auto judgement output must be a JSON array/object")

    parsed: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        keyword = str(item.get("keyword") or "").strip()
        if not keyword:
            continue
        decision = str(item.get("decision") or "skip_uncertain").strip().lower()
        if decision not in AUTO_JUDGEMENT_DECISIONS:
            decision = "skip_uncertain"
        entry_type = str(item.get("type") or ("nothing" if decision.startswith("skip_") else "confused")).strip().lower()
        if entry_type not in {"key", "black", "confused", "nothing"}:
            entry_type = "confused"
        value = str(item.get("value") or "").strip()
        if decision.startswith("skip_"):
            value = ""
            if entry_type not in {"confused", "nothing"}:
                entry_type = "nothing" if decision in {"skip_duplicate", "skip_noise"} else "confused"
        context_ids = item.get("context_ids", [])
        if not isinstance(context_ids, list):
            context_ids = [context_ids]
        matched_existing_sense_ids = item.get("matched_existing_sense_ids", [])
        if not isinstance(matched_existing_sense_ids, list):
            matched_existing_sense_ids = [matched_existing_sense_ids]
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = [aliases] if aliases else []
        parsed.append(
            {
                "keyword": keyword,
                "decision": decision,
                "type": entry_type,
                "value": value,
                "context_ids": [str(ctx_id).strip() for ctx_id in context_ids if str(ctx_id).strip()],
                "matched_existing_sense_ids": [str(sense_id).strip() for sense_id in matched_existing_sense_ids if str(sense_id).strip()],
                "aliases": [str(alias).strip() for alias in aliases if str(alias).strip()],
                "reason": str(item.get("reason") or "").strip(),
            }
        )
    return parsed


def sync_openclaw_judgements(
    *,
    output_dir: str | Path,
    judgements: list[dict[str, Any]],
    source: str,
    remote: bool,
    write_local: bool,
    force_remote_create: bool,
    client: Any | None = None,
    publishable_only: bool = False,
) -> dict[str, Any]:
    store = LingoStore(output_dir)
    upserted: list[dict[str, Any]] = []
    created_like = {"create_entry", "append_new_sense"}

    filtered_judgements = [
        item
        for item in judgements
        if not publishable_only or item.get("decision") in created_like
    ]
    remote_client = client if remote else None

    for item in filtered_judgements:
        decision = str(item.get("decision") or "").strip().lower()
        keyword = str(item.get("keyword") or "").strip()
        if decision not in created_like or not keyword:
            continue
        entry_type = str(item.get("type") or "nothing").strip().lower()
        value = str(item.get("value") or "").strip()
        aliases = [str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()]
        context_ids = [str(ctx_id).strip() for ctx_id in item.get("context_ids", []) if str(ctx_id).strip()]

        entity_id = ""
        remote_created: dict[str, Any] | None = None
        remote_create_skipped = False
        remote_skip_reason = ""
        if remote_client is not None and entry_type in {"key", "black"} and value:
            existing = store.get_entry(keyword)
            if (
                not force_remote_create
                and existing
                and any(
                    str(sense.get("value") or "").strip() == value
                    for sense in existing.get("senses", [])
                    if isinstance(sense, dict)
                )
            ):
                remote_create_skipped = True
                remote_skip_reason = "duplicate_guard: same keyword/value already exists in local senses"
            else:
                remote_created = remote_client.create_lingo_entity(
                    key=keyword,
                    description=value,
                    aliases=aliases,
                    provider="sofree-knowledge-cli",
                    outer_id=keyword,
                )
                entity_id = str(remote_created.get("entity_id") or "")
        elif remote_client is not None:
            remote_create_skipped = True
            remote_skip_reason = "remote_create_skipped: only key/black with non-empty value are written remotely"

        local_entry: dict[str, Any] | None = None
        if write_local:
            local_entry = store.upsert_entry(
                keyword=keyword,
                entry_type=entry_type,
                value=value,
                aliases=aliases,
                source=source,
                entity_id=entity_id,
                context_ids=context_ids,
                append_sense=(decision == "append_new_sense"),
            )

        upserted.append(
            {
                "keyword": keyword,
                "decision": decision,
                "type": entry_type,
                "value": value,
                "aliases": aliases,
                "entity_id": entity_id,
                "remote_created": remote_created,
                "remote_create_skipped": remote_create_skipped,
                "remote_create_skip_reason": remote_skip_reason,
                "entry": local_entry,
                "matched_existing_sense_ids": item.get("matched_existing_sense_ids", []),
                "reason": str(item.get("reason") or ""),
            }
        )

    created_entries = [
        {
            "keyword": item.get("keyword"),
            "type": item.get("type"),
            "value": item.get("value"),
            "context_ids": item.get("context_ids", []),
            "aliases": item.get("aliases", []),
        }
        for item in filtered_judgements
        if item.get("decision") in created_like
    ]
    publishable_entries = publishable_lingo_judgements(created_entries)

    return {
        "assume_success": True,
        "remote_enabled": bool(remote),
        "local_enabled": bool(write_local),
        "count": len(upserted),
        "entries": upserted,
        "publishable_count": len(publishable_entries),
        "lingo_store_file": str(store.path),
    }


def _classify_messages(messages: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    _ensure_project_root_on_syspath()
    from token_classify.classify import classify  # type: ignore

    normalized_messages = [
        {
            "message_id": str(item.get("message_id") or ""),
            "content": str(item.get("content") or item.get("text") or ""),
            "chat_id": str(item.get("chat_id") or ""),
            "create_time": str(item.get("create_time") or ""),
            "sender_name": str(item.get("sender_name") or ""),
        }
        for item in messages
        if str(item.get("content") or item.get("text") or "").strip()
    ]
    return classify(normalized_messages, config)


def _load_archive_messages(archive_dir: Path) -> list[dict[str, Any]]:
    messages_path = archive_dir / "messages.jsonl"
    if not messages_path.exists():
        return []
    items: list[dict[str, Any]] = []
    for raw_line in messages_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        items.append(
            {
                "message_id": str(payload.get("message_id") or ""),
                "chat_id": str(payload.get("chat_id") or ""),
                "create_time": str(payload.get("create_time") or ""),
                "content": str(payload.get("content") or payload.get("text") or ""),
                "text": str(payload.get("content") or payload.get("text") or ""),
                "sender_name": _resolve_sender_name(payload.get("sender")),
                "raw": payload,
            }
        )
    return items


def _find_related_existing_entries(store: LingoStore, keyword: str) -> list[dict[str, Any]]:
    normalized = str(keyword or "").strip().lower()
    related: list[dict[str, Any]] = []
    for entry in store.list_entries():
        entry_keyword = str(entry.get("keyword") or "").strip()
        entry_keyword_lower = entry_keyword.lower()
        aliases = [str(alias).strip().lower() for alias in entry.get("aliases", []) if str(alias).strip()]
        if entry_keyword_lower == normalized:
            continue
        if normalized in entry_keyword_lower or entry_keyword_lower in normalized or normalized in aliases:
            related.append(entry)
        if len(related) >= 5:
            break
    return related


def _compact_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    return {
        "keyword": str(entry.get("keyword") or ""),
        "aliases": [str(alias).strip() for alias in entry.get("aliases", []) if str(alias).strip()],
        "senses": [
            {
                "sense_id": str(sense.get("sense_id") or ""),
                "type": str(sense.get("type") or ""),
                "value": str(sense.get("value") or ""),
                "entity_id": str(sense.get("entity_id") or ""),
            }
            for sense in entry.get("senses", [])
            if isinstance(sense, dict)
        ],
    }


def _ensure_project_root_on_syspath() -> None:
    project_root = Path(__file__).resolve().parents[3]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def _resolve_time_window(*, recent_days: int, start_time: str, end_time: str, now: datetime) -> tuple[str, str]:
    normalized_end = _parse_datetime(end_time) or now
    normalized_start = _parse_datetime(start_time) or (normalized_end - timedelta(days=max(1, int(recent_days))))
    return (
        str(int(normalized_start.timestamp())),
        str(int(normalized_end.timestamp())),
    )


def _resolve_sender_name(sender: Any) -> str:
    if isinstance(sender, dict):
        return str(sender.get("name") or sender.get("display_name") or sender.get("id") or "")
    return ""


def _is_candidate_keyword(keyword: str) -> bool:
    normalized = str(keyword or "").strip()
    if len(normalized) < 2 or len(normalized) > 40:
        return False
    if normalized.isdigit():
        return False
    if normalized.lower() in {"好的", "收到", "今天", "明天", "现在", "这个", "那个"}:
        return False
    return True


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        timestamp = int(text)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
