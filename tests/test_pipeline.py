#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detailed smoke tests for token_classify.classify."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from message_extract.context_extractor import (
    DEFAULT_MESSAGE_WINDOW_SENTENCES,
    DEFAULT_TEXT_WINDOW_TOKENS,
)
from message_extract.extract_chat_messages import extract_plain_messages, load_records
from token_classify.classify import (
    _aggregate_sense_results,
    TextKeywordClassifierPipeline,
    classify,
)


def _show_stage(title: str, payload: Any, *, max_chars: int = 1200) -> None:
    print(f"\n=== {title} ===")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        print(text[:max_chars] + "\n...(truncated)...")
    else:
        print(text)


def test_default_windows() -> None:
    assert DEFAULT_MESSAGE_WINDOW_SENTENCES == 4, "message window should be +/-4 sentences"
    assert DEFAULT_TEXT_WINDOW_TOKENS == 80, "text window should be +/-80 tokens"
    _show_stage(
        "Default Window Config",
        {
            "message_window_sentences": DEFAULT_MESSAGE_WINDOW_SENTENCES,
            "text_window_tokens": DEFAULT_TEXT_WINDOW_TOKENS,
        },
    )


def _validate_result_schema(result: Dict[str, Any]) -> None:
    assert "statistics" in result
    assert "semantic_filter_details" in result
    assert "top_keywords" in result
    assert "keyword_contexts" in result
    assert "context_instances" in result
    assert "context_groups" in result
    assert "group_classification" in result
    assert "classification_results" in result

    for keyword in result["top_keywords"]:
        entries = result["classification_results"].get(keyword)
        assert isinstance(entries, list), "classification_results[keyword] must be a list"
        for item in entries:
            for field in ("type", "sense", "contexts", "count", "ratio"):
                assert field in item, f"missing field: {field}"


def _show_pipeline_details(case_name: str, result: Dict[str, Any]) -> None:
    _show_stage(f"{case_name} - 1) Statistics", result["statistics"])
    _show_stage(
        f"{case_name} - 2) Semantic Filter",
        {
            "threshold": result["semantic_filter_details"].get("threshold"),
            "meaningful_tokens_sample": result["semantic_filter_details"].get("meaningful_tokens", [])[:20],
            "meaningless_tokens_sample": result["semantic_filter_details"].get("meaningless_tokens", [])[:20],
            "filtered_tokens_sample": result["semantic_filter_details"].get("filtered_tokens", [])[:40],
        },
    )
    _show_stage(f"{case_name} - 3) Top Keywords", result["top_keywords"])
    _show_stage(
        f"{case_name} - 4) Keyword Contexts (sample)",
        {k: v[:2] for k, v in result["keyword_contexts"].items()},
    )
    _show_stage(
        f"{case_name} - 5) Context Instances (sample)",
        result["context_instances"][:5],
    )
    _show_stage(
        f"{case_name} - 6) Context Groups",
        result["context_groups"],
    )
    _show_stage(
        f"{case_name} - 7) Group Classification",
        result["group_classification"],
    )
    _show_stage(
        f"{case_name} - 8) Final classification_results",
        result["classification_results"],
    )


def test_end_to_end_text() -> None:
    text = (
        "AAAA领域研究深入。"
        "AAA与B协同落地。"
        "AAA面向C场景优化。"
        "BBB在系统集成中常见。"
        "CCC在数据处理中常见。"
    )
    result = classify(
        text,
        {
            "top_keywords": 3,
            "enable_analyzer": False,
            "classifier_config": {"enabled": False},
            "text_window_tokens": 80,
            "message_window_sentences": 4,
            "contexts_per_sense": 3,
        },
    )
    _validate_result_schema(result)
    assert result["statistics"]["text_window_tokens"] == 80
    assert result["statistics"]["message_window_sentences"] == 4
    _show_pipeline_details("TEXT INPUT", result)


def test_end_to_end_dialog_list() -> None:
    dialog = [
        "我们这周讨论 A 的落地方案，A 在推荐系统里效果不错。",
        "B 模块与 A 有耦合，需要拆分接口。",
        "A 在数据治理场景也出现了，和前面的语义不完全一样。",
        "C 主要是日志平台，和 A/B 都有关系。",
    ]
    result = classify(
        dialog,
        {
            "top_keywords": 4,
            "enable_analyzer": False,
            "classifier_config": {"enabled": False},
            "text_window_tokens": 80,
            "message_window_sentences": 4,
            "contexts_per_sense": 3,
        },
    )
    _validate_result_schema(result)
    assert result["statistics"]["total_messages"] == len(dialog)
    _show_pipeline_details("DIALOG INPUT", result)


def test_multi_sense_aggregation() -> None:
    top_keywords = ["A"]
    instances = [
        {"instance_id": "i1", "keyword": "A", "context": "A 在 场景 a1", "source_type": "text", "related_keywords": ["A"]},
        {"instance_id": "i2", "keyword": "A", "context": "A 在 场景 a2", "source_type": "text", "related_keywords": ["A"]},
        {"instance_id": "i3", "keyword": "A", "context": "A 在 场景 a3", "source_type": "text", "related_keywords": ["A"]},
        {"instance_id": "i4", "keyword": "A", "context": "A 在 场景 a4", "source_type": "text", "related_keywords": ["A"]},
        {"instance_id": "i5", "keyword": "A", "context": "A 在 场景 b1", "source_type": "text", "related_keywords": ["A"]},
        {"instance_id": "i6", "keyword": "A", "context": "A 在 场景 b2", "source_type": "text", "related_keywords": ["A"]},
        {"instance_id": "i7", "keyword": "A", "context": "A 在 场景 b3", "source_type": "text", "related_keywords": ["A"]},
        {"instance_id": "i8", "keyword": "A", "context": "A 在 场景 c1", "source_type": "text", "related_keywords": ["A"]},
        {"instance_id": "i9", "keyword": "A", "context": "A 在 场景 c2", "source_type": "text", "related_keywords": ["A"]},
        {"instance_id": "i10", "keyword": "A", "context": "A 在 场景 c3", "source_type": "text", "related_keywords": ["A"]},
    ]
    groups = [
        {"group_id": "g1", "keywords": ["A"], "contexts": ["..."], "instance_ids": ["i1", "i2", "i3", "i4"]},
        {"group_id": "g2", "keywords": ["A"], "contexts": ["..."], "instance_ids": ["i5", "i6", "i7"]},
        {"group_id": "g3", "keywords": ["A"], "contexts": ["..."], "instance_ids": ["i8", "i9", "i10"]},
    ]
    group_classification = {
        "g1": {"A": {"type": "key", "sense": "含义a"}},
        "g2": {"A": {"type": "key", "sense": "含义b"}},
        "g3": {"A": {"type": "key", "sense": "含义c"}},
    }
    aggregated = _aggregate_sense_results(
        top_keywords,
        instances,
        groups,
        group_classification,
        contexts_per_sense=3,
    )
    entries = aggregated["A"]
    total_count = sum(int(item["count"]) for item in entries)
    total_ratio = sum(float(item["ratio"]) for item in entries)
    assert len(entries) == 3
    assert total_count == 10
    assert 0.99 <= total_ratio <= 1.01
    _show_stage("Multi-sense Aggregation", aggregated)


def test_archive_reproducible_flow() -> None:
    """
    Reproducible flow requested by user:
    message_archive/messages.jsonl -> plain text -> classify
    Save text/tokens/filtered/classification to token_classify/outputs.
    """
    input_path = Path("message_archive/20260425T130609Z/messages.jsonl")
    assert input_path.exists(), f"input not found: {input_path}"

    records = load_records(input_path)
    plain_messages = extract_plain_messages(records, include_types={"text", "post"})[:1]
    plain_text = "\n".join(plain_messages)

    config = {
        "top_keywords": 20,
        "text_window_tokens": 80,
        "message_window_sentences": 4,
    }
    result = classify(plain_text, config)
    pipeline = TextKeywordClassifierPipeline(config)
    tokens = pipeline.tokenizer.tokenize(plain_text)
    filtered_tokens = result.get("semantic_filter_details", {}).get("filtered_tokens", [])
    classification_results = result.get("classification_results", {})

    out_dir = Path("token_classify/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "20260425T132724Z"
    output_paths = {
        "plain_text": out_dir / f"{prefix}_plain_text.txt",
        "tokens": out_dir / f"{prefix}_tokens.json",
        "filtered_tokens": out_dir / f"{prefix}_filtered_tokens.json",
        "classification_results": out_dir / f"{prefix}_classification_results.json",
    }

    output_paths["plain_text"].write_text(plain_text, encoding="utf-8")
    output_paths["tokens"].write_text(json.dumps(tokens, ensure_ascii=False, indent=2), encoding="utf-8")
    output_paths["filtered_tokens"].write_text(
        json.dumps(filtered_tokens, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_paths["classification_results"].write_text(
        json.dumps(classification_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for name, path in output_paths.items():
        assert path.exists(), f"{name} output not generated: {path}"

    _show_stage(
        "Archive Reproducible Flow Summary",
        {
            "input": str(input_path),
            "records_count": len(records),
            "plain_messages_count": len(plain_messages),
            "plain_text_chars": len(plain_text),
            "tokens_count": len(tokens),
            "filtered_tokens_count": len(filtered_tokens),
            "output_files": {k: str(v) for k, v in output_paths.items()},
        },
    )


def main() -> int:
    print("Running token_classify detailed pipeline tests...")
    # test_default_windows()
    # test_end_to_end_text()
    # test_end_to_end_dialog_list()
    # test_multi_sense_aggregation()
    test_archive_reproducible_flow()
    print("\n[PASS] all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
