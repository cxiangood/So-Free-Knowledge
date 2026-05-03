from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .dual_tower_dataset import build_weak_supervision_samples
from .two_tower import score_dual_tower_texts


URL_TOKEN_STOPWORDS = {
    "http",
    "https",
    "www",
    "feishu",
    "larksuite",
    "applink",
    "openapi",
    "openapis",
    "client",
    "chat",
    "openchatid",
    "openmessageid",
    "doc",
    "docx",
    "wiki",
    "sheet",
    "sheets",
    "base",
    "slides",
    "file",
    "files",
    "com",
    "cn",
}
MAX_CONTENT_BONUS = 12.0
DEFAULT_BONUS_SCALE_GRID = (0.0, 0.02, 0.05, 0.1, 0.2)
ROLE_FAMILY_HINTS: dict[str, tuple[str, ...]] = {
    "leader": ("leader", "负责人", "技术管理", "项目协调", "manager"),
    "algo": ("算法", "感知", "模型", "识别"),
    "data": ("数据", "etl", "采集", "标注"),
    "backend": ("后端", "接口", "服务", "缓存", "数据库"),
    "frontend": ("前端", "交互", "界面", "渲染", "展示"),
    "qa": ("测试", "验收", "回归", "复现"),
    "intern": ("实习", "助理", "研究支持", "调研", "文档"),
}
CONTENT_SIGNAL_HINTS: dict[str, tuple[str, ...]] = {
    "risk": ("风险", "异常", "故障", "阻塞", "回退", "降级", "失败"),
    "action": ("需要", "请", "负责", "跟进", "修复", "补充", "排查", "确认", "处理"),
    "knowledge": ("复盘", "总结", "经验", "文档", "沉淀", "结论", "方案"),
    "acceptance": ("验收", "回归", "验证", "标准", "case", "用例"),
    "data": ("数据", "字段", "链路", "采集", "标注", "口径", "日志"),
    "service": ("接口", "服务", "延迟", "缓存", "数据库", "部署", "任务"),
    "ux": ("界面", "展示", "交互", "渲染", "卡顿", "体验"),
}


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
    positive_pair_features: Counter[str] = Counter()
    negative_pair_features: Counter[str] = Counter()
    training_pairs = 0
    for sample in samples:
        strength = max(1, int(sample.get("strength", 1) or 1))
        user_text = str(sample.get("user_tower_text") or "")
        positive_text = str(sample.get("positive_doc_text") or "")
        for token in set(_tokenize_text(positive_text)):
            positive_counts[token] += strength
        for feature in set(_extract_pair_features(user_text, positive_text)):
            positive_pair_features[feature] += strength
        negatives = sample.get("negative_doc_texts", [])
        if isinstance(negatives, list):
            for negative_text in negatives:
                for token in set(_tokenize_text(str(negative_text or ""))):
                    negative_counts[token] += 1
                for feature in set(_extract_pair_features(user_text, str(negative_text or ""))):
                    negative_pair_features[feature] += 1
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

    all_pair_features = set(positive_pair_features) | set(negative_pair_features)
    pair_feature_weights: dict[str, float] = {}
    for feature in all_pair_features:
        pos = float(positive_pair_features.get(feature, 0))
        neg = float(negative_pair_features.get(feature, 0))
        weight = (pos + 1.0) / (neg + 1.0)
        score = round(weight - 1.0, 6)
        if score >= float(min_token_weight):
            pair_feature_weights[feature] = score

    bonus_scale = _select_bonus_scale(
        samples,
        token_weights=token_weights,
        pair_feature_weights=pair_feature_weights,
    )
    quality = evaluate_dual_tower_baseline(
        samples,
        token_weights=token_weights,
        pair_feature_weights=pair_feature_weights,
        bonus_scale=bonus_scale,
    )
    model = {
        "model_type": "dual_tower_baseline_term_weight",
        "sample_summary": summarize_dual_tower_samples(samples),
        "training_pairs": training_pairs,
        "bonus_scale": bonus_scale,
        "token_weights": dict(sorted(token_weights.items(), key=lambda item: item[1], reverse=True)[:5000]),
        "pair_feature_weights": dict(
            sorted(pair_feature_weights.items(), key=lambda item: item[1], reverse=True)[:500]
        ),
        "quality": quality,
    }
    result: dict[str, Any] = {
        "ok": True,
        "model_type": model["model_type"],
        "sample_summary": model["sample_summary"],
        "training_pairs": training_pairs,
        "bonus_scale": bonus_scale,
        "quality": quality,
        "top_tokens": list(model["token_weights"].items())[:20],
        "top_pair_features": list(model["pair_feature_weights"].items())[:20],
    }
    if str(output_file or "").strip():
        path = Path(output_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_file"] = str(path)
    else:
        result["model"] = model
    return result


def evaluate_dual_tower_baseline(
    samples: list[dict[str, Any]],
    *,
    token_weights: dict[str, float],
    pair_feature_weights: dict[str, float] | None = None,
    bonus_scale: float = 0.0,
) -> dict[str, Any]:
    hit_count = 0
    total = 0
    margin_sum = 0.0
    for sample in samples:
        user_text = str(sample.get("user_tower_text") or "")
        positive_text = str(sample.get("positive_doc_text") or "")
        negative_texts = [str(item or "") for item in sample.get("negative_doc_texts", []) if str(item or "").strip()]
        if not positive_text or not negative_texts:
            continue
        positive_score = _score_weighted_pair(
            user_text,
            positive_text,
            token_weights,
            pair_feature_weights=pair_feature_weights or {},
            bonus_scale=bonus_scale,
        )
        negative_scores = [
            _score_weighted_pair(
                user_text,
                text,
                token_weights,
                pair_feature_weights=pair_feature_weights or {},
                bonus_scale=bonus_scale,
            )
            for text in negative_texts
        ]
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
    pair_feature_weights_raw = model.get("pair_feature_weights", {}) if isinstance(model, dict) else {}
    pair_feature_weights = (
        {str(key): float(value) for key, value in pair_feature_weights_raw.items()}
        if isinstance(pair_feature_weights_raw, dict)
        else {}
    )
    bonus_scale = float(model.get("bonus_scale", 0.0) or 0.0) if isinstance(model, dict) else 0.0
    return _score_weighted_pair(
        user_text,
        content_text,
        token_weights,
        pair_feature_weights=pair_feature_weights,
        bonus_scale=bonus_scale,
    )


def score_dual_tower_bonus(content_text: str, model: dict[str, Any]) -> float:
    token_weights_raw = model.get("token_weights", {}) if isinstance(model, dict) else {}
    token_weights = (
        {str(key): float(value) for key, value in token_weights_raw.items()}
        if isinstance(token_weights_raw, dict)
        else {}
    )
    bonus_scale = float(model.get("bonus_scale", 0.0) or 0.0) if isinstance(model, dict) else 0.0
    return _content_bonus_score(content_text, token_weights, bonus_scale=bonus_scale)


def _score_weighted_pair(
    user_text: str,
    content_text: str,
    token_weights: dict[str, float],
    *,
    pair_feature_weights: dict[str, float] | None = None,
    bonus_scale: float = 0.0,
) -> float:
    base_score = score_dual_tower_texts(user_text, content_text)
    overlap_bonus = _overlap_bonus_score(user_text, content_text, token_weights, bonus_scale=bonus_scale)
    feature_bonus = _pair_feature_bonus_score(
        user_text,
        content_text,
        pair_feature_weights or {},
        bonus_scale=bonus_scale,
    )
    bonus = overlap_bonus + feature_bonus
    return base_score + bonus


def _content_bonus_score(content_text: str, token_weights: dict[str, float], *, bonus_scale: float = 0.0) -> float:
    content_tokens = set(_tokenize_text(content_text))
    if not content_tokens:
        return 0.0
    raw_bonus = sum(float(token_weights.get(token, 0.0)) for token in content_tokens)
    normalized_bonus = raw_bonus / math.sqrt(len(content_tokens))
    clipped_bonus = min(MAX_CONTENT_BONUS, normalized_bonus)
    # Only use model bonus as a positive supplement to cold-start.
    return max(0.0, float(bonus_scale) * clipped_bonus)


def _overlap_bonus_score(
    user_text: str,
    content_text: str,
    token_weights: dict[str, float],
    *,
    bonus_scale: float = 0.0,
) -> float:
    user_tokens = set(_tokenize_text(user_text))
    content_tokens = set(_tokenize_text(content_text))
    shared_tokens = user_tokens & content_tokens
    if not shared_tokens:
        return 0.0
    raw_bonus = sum(max(0.0, float(token_weights.get(token, 0.0))) for token in shared_tokens)
    normalized_bonus = raw_bonus / math.sqrt(len(shared_tokens))
    clipped_bonus = min(MAX_CONTENT_BONUS, normalized_bonus)
    return max(0.0, float(bonus_scale) * clipped_bonus)


def _select_bonus_scale(
    samples: list[dict[str, Any]],
    *,
    token_weights: dict[str, float],
    pair_feature_weights: dict[str, float] | None = None,
) -> float:
    best_scale = 0.0
    best_quality = evaluate_dual_tower_baseline(
        samples,
        token_weights=token_weights,
        pair_feature_weights=pair_feature_weights or {},
        bonus_scale=0.0,
    )
    for scale in DEFAULT_BONUS_SCALE_GRID[1:]:
        quality = evaluate_dual_tower_baseline(
            samples,
            token_weights=token_weights,
            pair_feature_weights=pair_feature_weights or {},
            bonus_scale=scale,
        )
        if quality["pairwise_hit_rate"] > best_quality["pairwise_hit_rate"]:
            best_quality = quality
            best_scale = float(scale)
            continue
        if (
            quality["pairwise_hit_rate"] == best_quality["pairwise_hit_rate"]
            and quality["avg_margin"] > best_quality["avg_margin"]
        ):
            best_quality = quality
            best_scale = float(scale)
    return best_scale


def _pair_feature_bonus_score(
    user_text: str,
    content_text: str,
    pair_feature_weights: dict[str, float],
    *,
    bonus_scale: float = 0.0,
) -> float:
    features = set(_extract_pair_features(user_text, content_text))
    if not features:
        return 0.0
    raw_bonus = sum(max(0.0, float(pair_feature_weights.get(feature, 0.0))) for feature in features)
    normalized_bonus = raw_bonus / math.sqrt(len(features))
    clipped_bonus = min(MAX_CONTENT_BONUS, normalized_bonus)
    return max(0.0, float(bonus_scale) * clipped_bonus)


def _extract_pair_features(user_text: str, content_text: str) -> list[str]:
    user_lower = str(user_text or "").lower()
    content_lower = str(content_text or "").lower()
    role_families = [family for family, hints in ROLE_FAMILY_HINTS.items() if any(hint in user_lower for hint in hints)]
    signal_families = [
        family for family, hints in CONTENT_SIGNAL_HINTS.items() if any(hint.lower() in content_lower for hint in hints)
    ]
    features: list[str] = []
    for role_family in role_families:
        features.append(f"role::{role_family}")
        for signal_family in signal_families:
            features.append(f"role::{role_family}__signal::{signal_family}")
    for signal_family in signal_families:
        features.append(f"signal::{signal_family}")
    return features


def _tokenize_text(text: str) -> list[str]:
    normalized = _normalize_text_for_tokenization(text)
    tokens = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    expanded: list[str] = []
    for token in tokens:
        if not _is_informative_token(token):
            continue
        expanded.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            for idx in range(0, len(token) - 1):
                bigram = token[idx : idx + 2]
                if _is_informative_token(bigram):
                    expanded.append(bigram)
    return expanded


def _normalize_text_for_tokenization(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"https?://\S+", " ", normalized)
    normalized = re.sub(r"\b[a-z0-9_-]+\.(?:com|cn|net|org)\b", " ", normalized)
    normalized = re.sub(r"[?&][a-z0-9_-]+=[^ \n|]+", " ", normalized)
    return normalized


def _is_informative_token(token: str) -> bool:
    value = str(token or "").strip().lower()
    if not value:
        return False
    if value in URL_TOKEN_STOPWORDS:
        return False
    if re.fullmatch(r"[a-z]+[0-9]{4,}[a-z0-9_]*", value):
        return False
    if re.fullmatch(r"[0-9_]+", value):
        return False
    if len(value) <= 2 and re.fullmatch(r"[a-z0-9_]+", value):
        return False
    return True
