from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .archive import collect_messages
from .lingo_context import extract_keyword_contexts, parse_lingo_judgements, publishable_lingo_judgements
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
    judgements = judge_lingo_candidates(candidates)
    publishable = publishable_lingo_judgements(judgements)

    run_dir = output_root / "lingo_auto_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "run_dir": str(run_dir),
        "candidates_file": str(run_dir / "candidates.json"),
        "judgements_file": str(run_dir / "judgements.json"),
        "publishable_file": str(run_dir / "publishable.json"),
        "classifier_result_file": str(run_dir / "classifier_result.json"),
    }
    (run_dir / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "judgements.json").write_text(json.dumps(judgements, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "publishable.json").write_text(json.dumps(publishable, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "classifier_result.json").write_text(
        json.dumps(classifier_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
            "last_publishable_count": len(publishable),
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
        "judgement_count": len(judgements),
        "publishable_count": len(publishable),
        "candidates": candidates,
        "judgements": judgements,
        "publishable_judgements": publishable,
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
    existing_keywords = {str(item.get("keyword") or "").strip() for item in store.list_entries()}
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
        if keyword in existing_keywords:
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


def judge_lingo_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    prompt = build_lingo_auto_judge_prompt(candidates)
    raw = _invoke_llm_judge(prompt)
    if raw:
        try:
            parsed = parse_lingo_judgements(raw)
        except Exception:
            parsed = []
        if parsed:
            candidate_map = {str(item.get("keyword") or ""): item for item in candidates}
            normalized: list[dict[str, Any]] = []
            for item in parsed:
                candidate = candidate_map.get(str(item.get("keyword") or ""))
                if candidate is None:
                    continue
                merged = dict(item)
                merged.setdefault("context_ids", list(candidate.get("context_ids", [])))
                aliases = item.get("aliases", [])
                if not isinstance(aliases, list):
                    aliases = []
                merged["aliases"] = [str(alias).strip() for alias in aliases if str(alias).strip()]
                normalized.append(merged)
            if normalized:
                return normalized

    return _heuristic_judgements(candidates)


def build_lingo_auto_judge_prompt(candidates: list[dict[str, Any]]) -> str:
    payload = {
        "task": "auto_select_new_lingo_entries",
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
        "你是 SoFree 的飞书词典自动审稿器。请根据候选词、近 7 天聊天上下文、分词频次和 BERT 语义信号，"
        "自动判断哪些词应该沉淀为新词条。只输出 JSON 数组，不要输出额外说明。\n"
        "每个元素必须包含: keyword, type, value, context_ids。\n"
        "可选字段: aliases。\n"
        "type 只能是 key / black / confused / nothing。\n"
        "规则:\n"
        "1. key: 值得沉淀的一般业务概念、流程名、指标名、系统名。\n"
        "2. black: 内部黑话、缩写、项目代号、默认外人看不懂的词。\n"
        "3. confused: 上下文不够，暂不入库。\n"
        "4. nothing: 噪音词、过泛词、口头语、普通动词，不入库，value 必须为空字符串。\n"
        "5. value 要写成适合飞书词典直接入库的简洁中文定义，尽量 12~40 字。\n"
        "6. aliases 仅在上下文里明确出现同义写法时给出。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _heuristic_judgements(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    judgements: list[dict[str, Any]] = []
    for item in candidates:
        keyword = str(item.get("keyword") or "").strip()
        initial_type = str(item.get("initial_type") or "confused").strip().lower()
        initial_value = str(item.get("initial_value") or "").strip()
        context_ids = [str(ctx_id) for ctx_id in item.get("context_ids", []) if str(ctx_id).strip()]
        frequency = int(item.get("frequency", 0) or 0)
        context_count = int(item.get("context_count", 0) or 0)
        if frequency < 2 or context_count < 1:
            judgement_type = "nothing"
            value = ""
        elif initial_type in {"key", "black"} and initial_value:
            judgement_type = initial_type
            value = initial_value
        elif _looks_like_black_term(keyword):
            judgement_type = "black"
            value = f"{keyword}：群聊中频繁出现的内部术语或缩写。"
        elif len(keyword) <= 2 and keyword.isascii():
            judgement_type = "nothing"
            value = ""
        else:
            judgement_type = "key"
            value = f"{keyword}：群聊中频繁讨论的业务概念，建议补充标准定义。"
        judgements.append(
            {
                "keyword": keyword,
                "type": judgement_type,
                "value": value if judgement_type != "nothing" else "",
                "context_ids": context_ids,
                "aliases": [],
            }
        )
    return judgements


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


def _invoke_llm_judge(prompt: str) -> str:
    try:
        _ensure_project_root_on_syspath()
        from llm.client import LLMClient, LLMConfig  # type: ignore
    except Exception:
        return ""

    config = LLMConfig.from_env(temperature=0.0, max_tokens=1800)
    if config.missing_fields():
        return ""
    client = LLMClient(config)
    return client.build_reply(
        "你是企业术语知识库自动审稿助手，只输出 JSON。",
        prompt,
    )


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


def _looks_like_black_term(keyword: str) -> bool:
    normalized = str(keyword or "").strip()
    return (
        any(ch.isupper() for ch in normalized)
        or any(ch.isdigit() for ch in normalized)
        or "/" in normalized
        or "-" in normalized
        or normalized.isascii()
    )


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
