from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .dual_tower_dataset import build_weak_supervision_samples
from .two_tower import score_dual_tower_texts


def export_dual_tower_samples(
    *,
    documents: list[dict[str, Any]],
    access_records: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    user_profile: dict[str, Any],
    target_user_id: str = "",
    output_file: str = "",
) -> dict[str, Any]:
    samples = build_weak_supervision_samples(
        documents=documents,
        access_records=access_records,
        messages=messages,
        user_profile=user_profile,
        target_user_id=target_user_id,
    )
    result: dict[str, Any] = {
        "sample_count": len(samples),
        "positive_count": len(samples),
        "has_output_file": bool(str(output_file or "").strip()),
        "samples": samples,
    }
    if str(output_file or "").strip():
        path = Path(output_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        result["output_file"] = str(path)
    return result


def append_dual_tower_samples(samples: list[dict[str, Any]], output_file: str) -> dict[str, Any]:
    path = Path(output_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return {
        "output_file": str(path),
        "appended_count": len(samples),
    }


def load_dual_tower_samples(input_file: str) -> list[dict[str, Any]]:
    path = Path(input_file).expanduser()
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                samples.append(loaded)
    return samples


def load_dual_tower_model(model_file: str) -> dict[str, Any]:
    path = Path(model_file).expanduser()
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    return loaded if isinstance(loaded, dict) else {}


def summarize_dual_tower_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    negative_counts = [len(sample.get("negative_doc_texts", [])) for sample in samples]
    strengths = [int(sample.get("strength", 1) or 1) for sample in samples]
    user_ids = {str(sample.get("user_id") or "").strip() for sample in samples if str(sample.get("user_id") or "").strip()}
    return {
        "sample_count": len(samples),
        "user_count": len(user_ids),
        "avg_negative_count": round(sum(negative_counts) / len(negative_counts), 3) if negative_counts else 0.0,
        "max_negative_count": max(negative_counts, default=0),
        "avg_strength": round(sum(strengths) / len(strengths), 3) if strengths else 0.0,
    }


def train_dual_tower_baseline(
    *,
    samples: list[dict[str, Any]],
    output_file: str = "",
    min_token_weight: float = 0.0,
) -> dict[str, Any]:
    positive_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    training_pairs = 0
    for sample in samples:
        strength = max(1, int(sample.get("strength", 1) or 1))
        positive_text = str(sample.get("positive_doc_text") or "")
        for token in _tokenize_text(positive_text):
            positive_counts[token] += strength
        negatives = sample.get("negative_doc_texts", [])
        if isinstance(negatives, list):
            for negative_text in negatives:
                for token in _tokenize_text(str(negative_text or "")):
                    negative_counts[token] += 1
                training_pairs += 1

    all_tokens = set(positive_counts) | set(negative_counts)
    token_weights: dict[str, float] = {}
    for token in all_tokens:
        pos = float(positive_counts.get(token, 0))
        neg = float(negative_counts.get(token, 0))
        weight = (pos + 1.0) / (neg + 1.0)
        score = round(weight - 1.0, 6)
        if score >= float(min_token_weight):
            token_weights[token] = score

    quality = evaluate_dual_tower_baseline(samples, token_weights=token_weights)
    model = {
        "model_type": "dual_tower_baseline_term_weight",
        "sample_summary": summarize_dual_tower_samples(samples),
        "training_pairs": training_pairs,
        "token_weights": dict(sorted(token_weights.items(), key=lambda item: item[1], reverse=True)[:5000]),
        "quality": quality,
    }
    result: dict[str, Any] = {
        "ok": True,
        "model_type": model["model_type"],
        "sample_summary": model["sample_summary"],
        "training_pairs": training_pairs,
        "quality": quality,
        "top_tokens": list(model["token_weights"].items())[:20],
    }
    if str(output_file or "").strip():
        path = Path(output_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_file"] = str(path)
    else:
        result["model"] = model
    return result


def evaluate_dual_tower_baseline(samples: list[dict[str, Any]], *, token_weights: dict[str, float]) -> dict[str, Any]:
    hit_count = 0
    total = 0
    margin_sum = 0.0
    for sample in samples:
        user_text = str(sample.get("user_tower_text") or "")
        positive_text = str(sample.get("positive_doc_text") or "")
        negative_texts = [str(item or "") for item in sample.get("negative_doc_texts", []) if str(item or "").strip()]
        if not positive_text or not negative_texts:
            continue
        positive_score = _score_weighted_pair(user_text, positive_text, token_weights)
        negative_scores = [_score_weighted_pair(user_text, text, token_weights) for text in negative_texts]
        if positive_score >= max(negative_scores):
            hit_count += 1
        margin_sum += positive_score - max(negative_scores)
        total += 1
    return {
        "evaluated_samples": total,
        "pairwise_hit_rate": round(hit_count / total, 4) if total else 0.0,
        "avg_margin": round(margin_sum / total, 6) if total else 0.0,
    }


def score_dual_tower_with_model(user_text: str, content_text: str, model: dict[str, Any]) -> float:
    token_weights_raw = model.get("token_weights", {}) if isinstance(model, dict) else {}
    token_weights = (
        {str(key): float(value) for key, value in token_weights_raw.items()}
        if isinstance(token_weights_raw, dict)
        else {}
    )
    return _score_weighted_pair(user_text, content_text, token_weights)


def _score_weighted_pair(user_text: str, content_text: str, token_weights: dict[str, float]) -> float:
    base_score = score_dual_tower_texts(user_text, content_text)
    bonus = 0.0
    for token in _tokenize_text(content_text):
        bonus += float(token_weights.get(token, 0.0))
    return base_score + bonus


def _tokenize_text(text: str) -> list[str]:
    import re

    normalized = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            for idx in range(0, len(token) - 1):
                expanded.append(token[idx : idx + 2])
    return expanded
