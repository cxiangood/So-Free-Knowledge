from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from insight.flow.engine import Engine, EngineConfig
from insight.msg.types import MessageEvent
from utils.logging_config import configure_logging, get_logger


DEFAULT_CSV_PATH = ROOT_DIR / "datas" / "chat_test_2.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "insight_module_eval"
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUT_DIR / "insight_full_pipeline_chat_test_2.jsonl"
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "insight_full_pipeline_chat_test_2_summary.json"
CONVERSATION_COLUMN = "连续对话（至少10轮）"
LOGGER = get_logger("insight.eval.full_pipeline_trace")


@dataclass(slots=True)
class EvalCase:
    case_id: str
    scenario: str
    conversation: list[str]


def load_cases(csv_path: Path) -> list[EvalCase]:
    LOGGER.info("Loading evaluation cases from csv: %s", csv_path)
    cases: list[EvalCase] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            raw = str(row.get(CONVERSATION_COLUMN, "") or "").strip()
            if not raw:
                continue
            case_id = str(row.get("用例ID", "") or "").strip() or f"case-{idx:03d}"
            scenario = str(row.get("场景名称", "") or "").strip()
            conversation = parse_conversation_lines(raw)
            if conversation:
                cases.append(EvalCase(case_id=case_id, scenario=scenario, conversation=conversation))
    LOGGER.info("Loaded %d cases from csv", len(cases))
    return cases


def parse_conversation_lines(raw_text: str) -> list[str]:
    rows: list[str] = []
    for line in raw_text.splitlines():
        text = str(line or "").strip()
        if not text:
            continue
        text = re.sub(r"^\d+\.\s*", "", text)
        rows.append(text)
    return rows


def parse_sender_and_content(line: str) -> tuple[str, str]:
    text = str(line or "").strip()
    match = re.match(r"^\[(?P<sender>[^\]]+)\]\s*(?P<content>.*)$", text)
    if match:
        return match.group("sender").strip(), match.group("content").strip()
    return "unknown", text


def build_message_event(*, case_id: str, round_index: int, sender: str, content: str) -> MessageEvent:
    message_id = f"{case_id}-m{round_index:02d}"
    chat_id = f"eval-chat-{case_id}"
    ts = f"2026-05-04T00:{round_index % 60:02d}:00Z"
    content_raw = json.dumps({"text": content}, ensure_ascii=False)
    return MessageEvent(
        event_type="im.message.receive_v1",
        event_id=f"evt-{message_id}",
        create_time=ts,
        message_id=message_id,
        chat_id=chat_id,
        chat_type="group",
        message_type="text",
        content_text=content,
        content_raw=content_raw,
        root_id="",
        parent_id="",
        update_time=ts,
        thread_id="",
        sender_open_id=sender,
        sender_union_id="",
        sender_user_id="",
        sender_type="user",
        tenant_key="",
        sender_name=sender,
        mentions=[],
        user_agent="",
        raw={
            "sender": {
                "sender_id": {"union_id": "", "user_id": "", "open_id": sender},
                "name": sender,
                "sender_type": "user",
                "tenant_key": "",
            },
            "message": {
                "message_id": message_id,
                "root_id": "",
                "parent_id": "",
                "create_time": ts,
                "update_time": ts,
                "chat_id": chat_id,
                "thread_id": "",
                "chat_type": "group",
                "message_type": "text",
                "content": content_raw,
                "mentions": [],
                "user_agent": "",
            },
        },
    )


def build_engine(output_dir: Path) -> Engine:
    run_tag = str(time.time_ns())
    state_dir = output_dir / f"state_full_pipeline_chat_test_2_{run_tag}"
    chat_history_path = state_dir / "chat_message_store.json"
    LOGGER.info("Building engine with isolated state_dir: %s", state_dir)
    config = EngineConfig(
        output_dir=output_dir,
        state_dir=state_dir,
        chat_history_path=chat_history_path,
        task_push_enabled=False,
        step_trace_enabled=False,
        rag_enabled=True,
        observe_auto_reply_enabled=True,
    )
    return Engine(config)


def normalize_modules(module_traces: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "detect": None,
        "lift": None,
        "route": None,
        "kb": [],
        "obs": [],
        "task": [],
    }
    if not isinstance(module_traces, dict):
        return base
    for key in base:
        if key in module_traces and module_traces[key] is not None:
            base[key] = module_traces[key]
    return base


def run_trace(csv_path: Path, output_jsonl: Path, summary_json: Path, max_cases: int | None = None) -> dict[str, Any]:
    LOGGER.info(
        "Starting full pipeline trace run, csv=%s, output_jsonl=%s, summary_json=%s, max_cases=%s",
        csv_path,
        output_jsonl,
        summary_json,
        max_cases,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases(csv_path)
    if max_cases is not None:
        cases = cases[: max(0, max_cases)]

    engine = build_engine(output_jsonl.parent)
    records: list[dict[str, Any]] = []
    route_counts: dict[str, int] = {}
    module_touched = {"detect": 0, "lift": 0, "route": 0, "kb": 0, "obs": 0, "task": 0}
    error_count = 0

    for case in tqdm(cases, desc="Running full insight trace", unit="case"):
        tqdm.write(f"[case] {case.case_id}")
        LOGGER.info("Case start: case_id=%s scenario=%s total_rounds=%d", case.case_id, case.scenario, len(case.conversation))
        if not case.conversation:
            LOGGER.warning("Case skipped due to empty conversation: case_id=%s", case.case_id)
            continue
        # Seed context first (all messages except the last one), then run once on the trigger message.
        seeded_count = 0
        for idx, line in enumerate(case.conversation[:-1], start=1):
            sender, content = parse_sender_and_content(line)
            seed_event = build_message_event(case_id=case.case_id, round_index=idx, sender=sender, content=content)
            # Seed chat context directly via existing message cache interface.
            try:
                engine.identity_map.update_from_event(seed_event)
            except Exception:
                pass
            engine.chat_store.append(seed_event)
            seeded_count += 1
        LOGGER.info("Case context seeded: case_id=%s seeded_messages=%d", case.case_id, seeded_count)

        trigger_round_index = len(case.conversation)
        trigger_line = case.conversation[-1]
        sender, content = parse_sender_and_content(trigger_line)
        event = build_message_event(case_id=case.case_id, round_index=trigger_round_index, sender=sender, content=content)
        LOGGER.info(
            "Running trigger message: case_id=%s message_id=%s round_index=%d sender=%s",
            case.case_id,
            event.message_id,
            trigger_round_index,
            sender,
        )
        result = engine.run(event, context={"mode": "eval_full_trace_trigger"})
        traces = normalize_modules(result.module_traces)

        if traces["detect"] is not None:
            module_touched["detect"] += 1
        if traces["lift"] is not None:
            module_touched["lift"] += 1
        if traces["route"] is not None:
            module_touched["route"] += 1
        if traces["kb"]:
            module_touched["kb"] += 1
        if traces["obs"]:
            module_touched["obs"] += 1
        if traces["task"]:
            module_touched["task"] += 1

        for target, count in result.routed_counts.items():
            route_counts[target] = route_counts.get(target, 0) + int(count)
        if result.errors:
            error_count += 1

        records.append(
            {
                "case_id": case.case_id,
                "scenario": case.scenario,
                "round_index": trigger_round_index,
                "message_id": event.message_id,
                "chat_id": event.chat_id,
                "sender": sender,
                "content": content,
                "context_messages": case.conversation[:-1],
                "detect": traces["detect"],
                "lift": traces["lift"],
                "route": traces["route"],
                "kb": traces["kb"],
                "obs": traces["obs"],
                "task": traces["task"],
                "errors": list(result.errors),
                "warnings": list(result.warnings),
                "skipped": bool(result.skipped),
                "created_at": result.created_at,
            }
        )
        LOGGER.info(
            "Case finished: case_id=%s message_id=%s skipped=%s errors=%d warnings=%d routed=%s",
            case.case_id,
            event.message_id,
            result.skipped,
            len(result.errors),
            len(result.warnings),
            result.routed_counts,
        )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote jsonl records: count=%d path=%s", len(records), output_jsonl)

    total = len(records)
    summary = {
        "csv_path": str(csv_path),
        "output_jsonl": str(output_jsonl),
        "total_cases": len(cases),
        "total_messages": total,
        "route_counts": route_counts,
        "module_touch_rate": {k: (v / total if total else 0.0) for k, v in module_touched.items()},
        "messages_with_errors": error_count,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote summary json: path=%s", summary_json)
    LOGGER.info("Run completed: total_cases=%d total_messages=%d error_messages=%d", len(cases), total, error_count)
    return summary


def main() -> None:
    configure_logging(level="INFO", app_name="INSIGHT_EVAL", force=False, quiet=True)
    parser = argparse.ArgumentParser(description="Run full insight pipeline per message and export module traces to JSONL.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to csv dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_JSON, help="Output summary JSON path")
    parser.add_argument("--max-cases", type=int, default=None, help="Only run first N cases for smoke test")
    args = parser.parse_args()

    summary = run_trace(csv_path=args.csv, output_jsonl=args.output, summary_json=args.summary, max_cases=args.max_cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
