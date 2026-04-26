#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-text keyword classification pipeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from message_extract.context_extractor import (
    DEFAULT_MESSAGE_WINDOW_SENTENCES,
    DEFAULT_TEXT_WINDOW_TOKENS,
    extract_keyword_context_instances_from_messages,
    extract_keyword_context_instances_from_text,
    group_context_instances,
)
from token_classify.domain_tokenizer import DomainAdaptiveTokenizer
from token_classify.token_filter import filter_by_semantic_metrics
from token_classify.word_frequency import calculate_word_frequency, get_top_keywords

try:
    from token_classify.keyword_classifier import KeywordClassifier
except Exception:  # pragma: no cover - optional dependency
    KeywordClassifier = None  # type: ignore[assignment]

try:
    from token_classify.analyzer import SemanticDensityAnalyzer
except Exception:  # pragma: no cover - optional dependency
    SemanticDensityAnalyzer = None  # type: ignore[assignment]


class _FallbackTokenizer:
    _token_re = __import__("re").compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_.+-]+")

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return self._token_re.findall(text)


def _aggregate_metrics(words: List[str], semantic_values: List[float], entropy_values: List[float]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "sum_sd": 0.0, "sum_ae": 0.0})
    for word, sd, ae in zip(words, semantic_values, entropy_values):
        key = " ".join(str(word).split())
        if not key:
            continue
        rec = stats[key]
        rec["count"] += 1.0
        rec["sum_sd"] += float(sd)
        rec["sum_ae"] += float(ae)

    out: Dict[str, Dict[str, float]] = {}
    for word, rec in stats.items():
        count = rec["count"] or 1.0
        out[word] = {
            "semantic_density": rec["sum_sd"] / count,
            "attention_entropy": rec["sum_ae"] / count,
        }
    return out


def _fallback_metrics(tokens: List[str]) -> Dict[str, Dict[str, float]]:
    counts = Counter(tokens)
    max_count = max(counts.values()) if counts else 1
    out: Dict[str, Dict[str, float]] = {}
    for token, count in counts.items():
        sd = float(count) / float(max_count)
        ae = max(0.0, 1.0 - sd)
        out[token] = {"semantic_density": sd, "attention_entropy": ae}
    return out


def _build_keyword_contexts(instances: List[Dict[str, object]], keywords: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {kw: [] for kw in keywords}
    for item in instances:
        kw = str(item.get("keyword", ""))
        if kw not in grouped:
            continue
        ctx = str(item.get("context", "")).strip()
        if ctx:
            grouped[kw].append(ctx)
    return grouped


def _aggregate_sense_results(
    top_keywords: List[str],
    instances: List[Dict[str, object]],
    groups: List[Dict[str, object]],
    group_classification: Dict[str, Dict[str, Dict[str, str]]],
    *,
    contexts_per_sense: int = 3,
) -> Dict[str, List[Dict[str, object]]]:
    instance_to_group: Dict[str, str] = {}
    for group in groups:
        group_id = str(group.get("group_id", ""))
        for inst_id in group.get("instance_ids", []):
            instance_to_group[str(inst_id)] = group_id

    per_keyword_instances: Dict[str, List[Dict[str, object]]] = {kw: [] for kw in top_keywords}
    for inst in instances:
        kw = str(inst.get("keyword", ""))
        if kw in per_keyword_instances:
            per_keyword_instances[kw].append(inst)

    final_results: Dict[str, List[Dict[str, object]]] = {}
    for kw in top_keywords:
        items = per_keyword_instances.get(kw, [])
        total = len(items)
        if total == 0:
            final_results[kw] = []
            continue

        agg: Dict[tuple, Dict[str, object]] = {}
        for inst in items:
            inst_id = str(inst.get("instance_id", ""))
            context = str(inst.get("context", "")).strip()
            group_id = instance_to_group.get(inst_id, "")
            pred = group_classification.get(group_id, {}).get(kw, {"type": "confused", "sense": "无分组分类结果"})
            item_type = str(pred.get("type", "confused"))
            sense = str(pred.get("sense", ""))
            key = (item_type, sense)
            if key not in agg:
                agg[key] = {
                    "type": item_type,
                    "sense": sense,
                    "contexts": [],
                    "count": 0,
                }
            agg[key]["count"] = int(agg[key]["count"]) + 1
            ctxs = agg[key]["contexts"]
            if isinstance(ctxs, list) and context and context not in ctxs and len(ctxs) < contexts_per_sense:
                ctxs.append(context)

        sense_list = sorted(agg.values(), key=lambda x: int(x["count"]), reverse=True)
        for item in sense_list:
            item["ratio"] = float(item["count"]) / float(total) if total else 0.0
        final_results[kw] = sense_list

    return final_results


class TextKeywordClassifierPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.top_keywords = int(self.config.get("top_keywords", 20))
        self.stop_words = self.config.get("stop_words", [])
        self.message_window_sentences = int(self.config.get("message_window_sentences", DEFAULT_MESSAGE_WINDOW_SENTENCES))
        self.text_window_tokens = int(self.config.get("text_window_tokens", DEFAULT_TEXT_WINDOW_TOKENS))
        self.contexts_per_sense = int(self.config.get("contexts_per_sense", 3))

        try:
            self.tokenizer = DomainAdaptiveTokenizer(**self.config.get("tokenizer_config", {}))
        except Exception:
            self.tokenizer = _FallbackTokenizer()
        self.enable_analyzer = bool(self.config.get("enable_analyzer", True))
        self.analyzer = None
        if self.enable_analyzer and SemanticDensityAnalyzer is not None:
            try:
                self.analyzer = SemanticDensityAnalyzer(**self.config.get("analyzer_config", {}))
            except Exception:
                self.analyzer = None

        self.classifier = None
        classifier_config = dict(self.config.get("classifier_config", {}))
        if classifier_config.get("enabled", True) and KeywordClassifier is not None:
            try:
                self.classifier = KeywordClassifier(**{k: v for k, v in classifier_config.items() if k != "enabled"})
            except Exception:
                self.classifier = None

    def _build_metrics(self, text: str, tokens: List[str]) -> Dict[str, Dict[str, float]]:
        if self.analyzer is None:
            return _fallback_metrics(tokens)
        try:
            words, semantic_values = self.analyzer.semantic_density(text)
            words2, entropy_values = self.analyzer.attention_entropy(text)
            if words != words2:
                return _fallback_metrics(tokens)
            metrics = _aggregate_metrics(words, semantic_values, entropy_values)
            return metrics if metrics else _fallback_metrics(tokens)
        except Exception:
            return _fallback_metrics(tokens)

    def _classify_groups(self, groups: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, str]]]:
        if not groups:
            return {}
        if self.classifier is None:
            out: Dict[str, Dict[str, Dict[str, str]]] = {}
            for group in groups:
                group_id = str(group.get("group_id", ""))
                kws = [str(x) for x in group.get("keywords", []) if str(x)]
                out[group_id] = {kw: {"type": "confused", "sense": "未配置LLM分类器"} for kw in kws}
            return out
        return self.classifier.batch_classify_with_groups(groups)

    def classify_text(self, text: str) -> Dict[str, Any]:
        tokens = self.tokenizer.tokenize(text)
        metrics = self._build_metrics(text, tokens)
        filter_details = filter_by_semantic_metrics(tokens, metrics, return_details=True)
        filtered_tokens = filter_details["filtered_tokens"]
        word_frequency = calculate_word_frequency(filtered_tokens, stop_words=self.stop_words)
        top_keywords = get_top_keywords(filtered_tokens, top_k=self.top_keywords, stop_words=self.stop_words)

        instances = extract_keyword_context_instances_from_text(
            text,
            top_keywords,
            window_tokens=self.text_window_tokens,
            tokenizer=self.tokenizer,
        )
        groups = group_context_instances(instances)
        group_classification = self._classify_groups(groups)
        classification_results = _aggregate_sense_results(
            top_keywords,
            instances,
            groups,
            group_classification,
            contexts_per_sense=self.contexts_per_sense,
        )

        return {
            "statistics": {
                "total_tokens": len(tokens),
                "filtered_tokens": len(filtered_tokens),
                "unique_words": len(word_frequency),
                "text_window_tokens": self.text_window_tokens,
                "message_window_sentences": self.message_window_sentences,
            },
            "word_frequency": word_frequency,
            "top_keywords": top_keywords,
            "keyword_contexts": _build_keyword_contexts(instances, top_keywords),
            "context_instances": instances,
            "context_groups": groups,
            "group_classification": group_classification,
            "classification_results": classification_results,
            "semantic_filter_details": filter_details,
        }

    @staticmethod
    def _normalize_dialog(dialog: Sequence[Any]) -> List[dict]:
        normalized: List[dict] = []
        for item in dialog:
            if isinstance(item, str):
                content = item.strip()
                if content:
                    normalized.append({"content": content})
                continue
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str) and content.strip():
                    normalized.append({"content": content.strip()})
        return normalized

    def classify_messages(self, messages: Sequence[Any]) -> Dict[str, Any]:
        normalized_messages = self._normalize_dialog(messages)
        text_parts: List[str] = []
        for msg in normalized_messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    text_parts.append(content)
        joined_text = "\n".join(text_parts)
        tokens = self.tokenizer.tokenize(joined_text)
        metrics = self._build_metrics(joined_text, tokens)
        filter_details = filter_by_semantic_metrics(tokens, metrics, return_details=True)
        filtered_tokens = filter_details["filtered_tokens"]
        word_frequency = calculate_word_frequency(filtered_tokens, stop_words=self.stop_words)
        top_keywords = get_top_keywords(filtered_tokens, top_k=self.top_keywords, stop_words=self.stop_words)

        instances = extract_keyword_context_instances_from_messages(
            normalized_messages,
            top_keywords,
            window_sentences=self.message_window_sentences,
        )
        groups = group_context_instances(instances)
        group_classification = self._classify_groups(groups)
        classification_results = _aggregate_sense_results(
            top_keywords,
            instances,
            groups,
            group_classification,
            contexts_per_sense=self.contexts_per_sense,
        )

        return {
            "statistics": {
                "total_messages": len(normalized_messages),
                "total_tokens": len(tokens),
                "filtered_tokens": len(filtered_tokens),
                "unique_words": len(word_frequency),
                "text_window_tokens": self.text_window_tokens,
                "message_window_sentences": self.message_window_sentences,
            },
            "word_frequency": word_frequency,
            "top_keywords": top_keywords,
            "keyword_contexts": _build_keyword_contexts(instances, top_keywords),
            "context_instances": instances,
            "context_groups": groups,
            "group_classification": group_classification,
            "classification_results": classification_results,
            "semantic_filter_details": filter_details,
        }


def classify_text(text: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return TextKeywordClassifierPipeline(config).classify_text(text)


def classify(input_data: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Unified entrypoint:
    - str: plain text
    - list: dialog list from extract_chat_messages plain outputs (list[str]) or list[{"content": "..."}]
    """
    pipeline = TextKeywordClassifierPipeline(config)
    if isinstance(input_data, str):
        return pipeline.classify_text(input_data)
    if isinstance(input_data, list):
        return pipeline.classify_messages(input_data)
    raise TypeError("input_data must be str or list")


def _load_text_from_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_dialog_from_file(path: str) -> List[Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--dialog-file must be a JSON array")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Long-text keyword classification")
    parser.add_argument("--text", help="Input long text")
    parser.add_argument("--text-file", help="Path to input text file")
    parser.add_argument("--dialog-file", help="Path to dialog json list (list[str] or list[{content:...}])")
    parser.add_argument("-o", "--output", help="Optional output json path")
    parser.add_argument("--top-keywords", type=int, default=20)
    parser.add_argument("--text-window-tokens", type=int, default=DEFAULT_TEXT_WINDOW_TOKENS)
    parser.add_argument("--message-window-sentences", type=int, default=DEFAULT_MESSAGE_WINDOW_SENTENCES)
    args = parser.parse_args()

    config = {
        "top_keywords": args.top_keywords,
        "text_window_tokens": args.text_window_tokens,
        "message_window_sentences": args.message_window_sentences,
    }
    if args.dialog_file:
        dialog = _load_dialog_from_file(args.dialog_file)
        result = classify(dialog, config)
    else:
        text = args.text or ""
        if not text and args.text_file:
            text = _load_text_from_file(args.text_file)
        if not text.strip():
            raise ValueError("Please provide --text/--text-file or --dialog-file")
        result = classify(text, config)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
