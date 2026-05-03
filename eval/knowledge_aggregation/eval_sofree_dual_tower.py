from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import zipfile
import hashlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}

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
RANDOM_REPEATS = 10
SEED = 42
LLM_SAMPLE_CASES = 6
ALPHA_GRID = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
SYNONYM_PAIRS = {
    "点云": "三维点云",
    "漏检": "漏识别",
    "回归集": "回放验证集",
    "阈值": "门限",
    "渲染": "绘制",
    "卡顿": "掉帧",
    "接口": "服务接口",
    "延迟": "时延",
    "降级策略": "兜底方案",
    "演示": "Demo",
    "采集链路": "数据采集流程",
    "缺列": "字段缺失",
    "风险": "隐患",
    "方案": "策略",
    "任务": "待办动作",
    "复盘": "经验总结",
    "知识沉淀": "知识归档",
    "验收": "验证通过",
    "召回率": "检出率",
    "目标检测": "目标识别",
}


@dataclass
class QueryCase:
    scene_id: str
    role_id: str
    role_name: str
    query_text: str
    positive_card_id: str
    positive_card_text: str
    negative_card_ids: list[str]
    negative_card_texts: list[str]
    no_value_text: str


@dataclass
class LLMJudgeConfig:
    api_key: str
    model_id: str
    base_url: str
    temperature: float = 0.0
    max_tokens: int = 300


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(0, value - 1)


def parse_xlsx(path: Path) -> dict[str, list[dict[str, str]]]:
    workbook_bytes = path.read_bytes()
    with zipfile.ZipFile(BytesIO(workbook_bytes)) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", NS):
                text = "".join(
                    node.text or ""
                    for node in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                )
                shared_strings.append(text)

        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root.findall("pkg:Relationship", NS)}

        parsed: dict[str, list[dict[str, str]]] = {}
        sheets = wb_root.find("main:sheets", NS)
        if sheets is None:
            return parsed

        for sheet in sheets.findall("main:sheet", NS):
            name = sheet.attrib.get("name", "")
            rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
            target = rel_map.get(rel_id, "").lstrip("/")
            if not target:
                continue
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            rows = parse_sheet_rows(zf.read(sheet_path), shared_strings)
            if not rows:
                parsed[name] = []
                continue
            headers = [str(item).strip() for item in rows[0]]
            records: list[dict[str, str]] = []
            for row in rows[1:]:
                if not any(str(item).strip() for item in row):
                    continue
                record: dict[str, str] = {}
                width = max(len(headers), len(row))
                for idx in range(width):
                    key = headers[idx] if idx < len(headers) and headers[idx] else f"col_{idx + 1}"
                    value = row[idx] if idx < len(row) else ""
                    record[key] = str(value).strip()
                records.append(record)
            parsed[name] = records
        return parsed


def parse_sheet_rows(raw_xml: bytes, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(raw_xml)
    sheet_data = root.find("main:sheetData", NS)
    if sheet_data is None:
        return []
    rows: list[list[str]] = []
    for row in sheet_data.findall("main:row", NS):
        values: list[str] = []
        current_col = 0
        for cell in row.findall("main:c", NS):
            ref = cell.attrib.get("r", "")
            col = column_index(ref) if ref else current_col
            while len(values) < col:
                values.append("")
            values.append(cell_value(cell, shared_strings))
            current_col = len(values)
        rows.append(values)
    return rows


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    value_node = cell.find("main:v", NS)
    inline_node = cell.find("main:is", NS)
    if cell_type == "s" and value_node is not None and value_node.text is not None:
        idx = int(value_node.text)
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
    if cell_type == "inlineStr" and inline_node is not None:
        return "".join(
            node.text or ""
            for node in inline_node.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        )
    if value_node is not None and value_node.text is not None:
        return value_node.text
    return ""


def normalize_dash(value: str) -> str:
    normalized = str(value or "").strip()
    return "" if normalized in {"—", "-", "无", "None", "none"} else normalized


def maybe_rewrite_with_synonyms(text: str, *, salt: str) -> str:
    rewritten = str(text or "")
    for source, target in SYNONYM_PAIRS.items():
        marker = stable_int(f"{salt}|{source}") % 100
        if source in rewritten and marker < 45:
            rewritten = rewritten.replace(source, target)
    return rewritten


def build_query_text(role: dict[str, str], scene: dict[str, str]) -> str:
    parts = [
        f"role_name: {role.get('姓名', '')}",
        f"role_title: {role.get('身份/岗位', '')}",
        f"profile: {role.get('画像', '')}",
        f"interests: {role.get('兴趣关键词', '')}",
        f"push_focus: {role.get('推送判断重点', '')}",
        f"business: {scene.get('业务域', '')}",
        f"scene_title: {scene.get('场景标题', '')}",
        f"scene_type: {scene.get('类型', '')}",
        f"diff_tags: {scene.get('差异标签', '')}",
        f"chat: {scene.get('聊天片段', '')}",
    ]
    return " | ".join(part for part in parts if part and not part.endswith(": "))


def build_card_text(card: dict[str, str]) -> str:
    parts = [
        f"business: {maybe_rewrite_with_synonyms(card.get('业务域', ''), salt=str(card.get('卡片ID', '')) + '|biz')}",
        f"scene_title: {maybe_rewrite_with_synonyms(card.get('场景标题', ''), salt=str(card.get('卡片ID', '')) + '|title')}",
        f"content: {maybe_rewrite_with_synonyms(card.get('卡片内容', ''), salt=str(card.get('卡片ID', '')) + '|content')}",
        f"hit_keywords: {maybe_rewrite_with_synonyms(card.get('命中关键词/兴趣', ''), salt=str(card.get('卡片ID', '')) + '|kw')}",
        f"why_push: {maybe_rewrite_with_synonyms(card.get('为什么值得推送', ''), salt=str(card.get('卡片ID', '')) + '|why')}",
        f"pool: {card.get('建议目标池', '')}",
    ]
    return " | ".join(part for part in parts if part and not part.endswith(": "))


def build_cases(data: dict[str, list[dict[str, str]]]) -> tuple[list[QueryCase], dict[str, str]]:
    roles = data["角色库"]
    scenes = {row["场景ID"]: row for row in data["50组聊天场景"]}
    cards = data["推送卡片明细"]
    no_value_map = {row["场景ID"]: row["不应推送的消息"] for row in data["无价值消息样本"]}
    role_by_name = {row["姓名"]: row for row in roles}

    card_text_by_id = {card["卡片ID"]: build_card_text(card) for card in cards}
    cards_by_scene_role = {(card["场景ID"], card["推送对象"]): card for card in cards}
    cards_by_scene: dict[str, list[dict[str, str]]] = {}
    for card in cards:
        cards_by_scene.setdefault(card["场景ID"], []).append(card)

    cases: list[QueryCase] = []
    for card in cards:
        scene_id = card["场景ID"]
        role_name = card["推送对象"]
        role = role_by_name[role_name]
        scene = scenes[scene_id]
        query_text = build_query_text(role, scene)
        positive_card_id = card["卡片ID"]
        positive_card_text = card_text_by_id[positive_card_id]
        same_scene_negatives = [item for item in cards_by_scene.get(scene_id, []) if item["卡片ID"] != positive_card_id]
        cross_scene_same_role = [item for item in cards if item["推送对象"] == role_name and item["卡片ID"] != positive_card_id]
        cross_scene_same_business = [
            item
            for item in cards
            if item["业务域"] == card["业务域"] and item["卡片ID"] != positive_card_id and item["推送对象"] != role_name
        ]
        negatives: list[dict[str, str]] = []
        seen: set[str] = set()
        for bucket in (same_scene_negatives, cross_scene_same_role, cross_scene_same_business):
            for item in bucket:
                cid = item["卡片ID"]
                if cid not in seen:
                    negatives.append(item)
                    seen.add(cid)
        negative_card_ids = [item["卡片ID"] for item in negatives]
        negative_card_texts = [card_text_by_id[cid] for cid in negative_card_ids]
        no_value_text = normalize_dash(no_value_map.get(scene_id, ""))
        cases.append(
            QueryCase(
                scene_id=scene_id,
                role_id=role.get("角色ID", ""),
                role_name=role_name,
                query_text=query_text,
                positive_card_id=positive_card_id,
                positive_card_text=positive_card_text,
                negative_card_ids=negative_card_ids,
                negative_card_texts=negative_card_texts,
                no_value_text=no_value_text,
            )
        )
    return cases, card_text_by_id


def normalize_text_for_tokenization(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"https?://\S+", " ", normalized)
    normalized = re.sub(r"\b[a-z0-9_-]+\.(?:com|cn|net|org)\b", " ", normalized)
    normalized = re.sub(r"[?&][a-z0-9_-]+=[^ \n|]+", " ", normalized)
    return normalized


def is_informative_token(token: str) -> bool:
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


@lru_cache(maxsize=4096)
def tokenize(text: str) -> tuple[str, ...]:
    normalized = normalize_text_for_tokenization(text)
    tokens = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    expanded: list[str] = []
    for token in tokens:
        if not is_informative_token(token):
            continue
        expanded.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            for idx in range(0, len(token) - 1):
                bigram = token[idx : idx + 2]
                if is_informative_token(bigram):
                    expanded.append(bigram)
    return tuple(expanded)


@lru_cache(maxsize=4096)
def text_to_vector(text: str) -> dict[str, float]:
    counts = Counter(tokenize(text))
    total = float(sum(counts.values()) or 1.0)
    return {token: count / total for token, count in counts.items()}


def cosine_score(user_text: str, content_text: str) -> float:
    user_vec = text_to_vector(user_text)
    content_vec = text_to_vector(content_text)
    if not user_vec or not content_vec:
        return 0.0
    numerator = sum(user_vec[token] * content_vec.get(token, 0.0) for token in user_vec)
    user_norm = math.sqrt(sum(value * value for value in user_vec.values()))
    content_norm = math.sqrt(sum(value * value for value in content_vec.values()))
    if user_norm <= 0.0 or content_norm <= 0.0:
        return 0.0
    return numerator / (user_norm * content_norm)


def train_token_weights(train_cases: list[QueryCase]) -> dict[str, float]:
    positive_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    for case in train_cases:
        for token in set(tokenize(case.positive_card_text)):
            positive_counts[token] += 1
        for text in case.negative_card_texts:
            for token in set(tokenize(text)):
                negative_counts[token] += 1
        if case.no_value_text:
            for token in set(tokenize(case.no_value_text)):
                negative_counts[token] += 1
    token_weights: dict[str, float] = {}
    for token in set(positive_counts) | set(negative_counts):
        pos = float(positive_counts.get(token, 0))
        neg = float(negative_counts.get(token, 0))
        token_weights[token] = round((pos + 1.0) / (neg + 1.0) - 1.0, 6)
    return token_weights


def weighted_score(user_text: str, content_text: str, token_weights: dict[str, float]) -> float:
    base_score = cosine_score(user_text, content_text)
    return base_score + content_bonus(content_text, token_weights)


def content_bonus(content_text: str, token_weights: dict[str, float]) -> float:
    content_tokens = set(tokenize(content_text))
    if not content_tokens:
        return 0.0
    raw_bonus = sum(float(token_weights.get(token, 0.0)) for token in content_tokens)
    normalized_bonus = raw_bonus / math.sqrt(len(content_tokens))
    return min(MAX_CONTENT_BONUS, normalized_bonus)


def fused_score(user_text: str, content_text: str, token_weights: dict[str, float], alpha: float) -> float:
    positive_bonus = max(0.0, content_bonus(content_text, token_weights))
    return cosine_score(user_text, content_text) + alpha * positive_bonus


def select_alpha(train_cases: list[QueryCase], card_text_by_id: dict[str, str], token_weights: dict[str, float]) -> float:
    best_alpha = 0.0
    best_mrr = -1.0
    for alpha in ALPHA_GRID:
        ranks: list[int] = []
        for case in train_cases:
            candidate_ids = [case.positive_card_id] + case.negative_card_ids
            scores = {
                cid: fused_score(case.query_text, card_text_by_id[cid], token_weights, alpha)
                for cid in candidate_ids
            }
            ranks.append(rank_of_positive(scores, case.positive_card_id, random.Random(SEED)))
        metrics = summarize_ranks(ranks)
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_alpha = alpha
    return best_alpha


def rank_of_positive(scores: dict[str, float], positive_id: str, rng: random.Random | None = None) -> int:
    tie_rng = rng or random.Random(SEED)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], tie_rng.random()))
    for idx, (candidate_id, _) in enumerate(ordered, start=1):
        if candidate_id == positive_id:
            return idx
    return len(ordered) + 1


def summarize_ranks(ranks: list[int]) -> dict[str, float]:
    total = len(ranks)
    return {
        "count": total,
        "hit@1": round(sum(1 for rank in ranks if rank <= 1) / total, 4),
        "hit@3": round(sum(1 for rank in ranks if rank <= 3) / total, 4),
        "hit@5": round(sum(1 for rank in ranks if rank <= 5) / total, 4),
        "mrr": round(sum(1.0 / rank for rank in ranks) / total, 4),
        "mean_rank": round(sum(ranks) / total, 3),
        "median_rank": round(statistics.median(ranks), 3),
    }


def stable_int(text: str) -> int:
    digest = hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def pairwise_accuracy(
    cases: list[QueryCase],
    scorer: Any,
) -> dict[str, float]:
    hard_win = 0
    no_value_win = 0
    hard_margin_sum = 0.0
    no_value_margin_sum = 0.0
    hard_total = 0
    no_value_total = 0
    for case in cases:
        pos_score = scorer(case.query_text, case.positive_card_text)
        if case.negative_card_texts:
            neg_scores = [scorer(case.query_text, text) for text in case.negative_card_texts]
            hard_total += 1
            hard_margin = pos_score - max(neg_scores)
            hard_margin_sum += hard_margin
            if hard_margin > 0:
                hard_win += 1
        if case.no_value_text:
            no_value_total += 1
            no_value_score = scorer(case.query_text, case.no_value_text)
            margin = pos_score - no_value_score
            no_value_margin_sum += margin
            if margin > 0:
                no_value_win += 1
    return {
        "hard_negative_pairwise_acc": round(hard_win / hard_total, 4) if hard_total else 0.0,
        "hard_negative_avg_margin": round(hard_margin_sum / hard_total, 4) if hard_total else 0.0,
        "no_value_pairwise_acc": round(no_value_win / no_value_total, 4) if no_value_total else 0.0,
        "no_value_avg_margin": round(no_value_margin_sum / no_value_total, 4) if no_value_total else 0.0,
    }


def evaluate(cases: list[QueryCase], card_text_by_id: dict[str, str]) -> dict[str, Any]:
    scene_ids = sorted({case.scene_id for case in cases})
    folds = [scene_ids[idx::5] for idx in range(5)]
    cold_ranks: list[int] = []
    dual_ranks: list[int] = []
    random_rank_runs: list[list[int]] = [[] for _ in range(RANDOM_REPEATS)]
    cold_pairwise_cases: list[QueryCase] = []
    dual_pairwise_cases: list[QueryCase] = []
    for fold_idx, test_scene_ids in enumerate(folds):
        test_scene_set = set(test_scene_ids)
        train_cases = [case for case in cases if case.scene_id not in test_scene_set]
        test_cases = [case for case in cases if case.scene_id in test_scene_set]
        token_weights = train_token_weights(train_cases)
        cold_pairwise_cases.extend(test_cases)
        dual_pairwise_cases.extend(test_cases)
        for case in test_cases:
            candidate_ids = [case.positive_card_id] + case.negative_card_ids
            candidate_scores_cold = {
                cid: cosine_score(case.query_text, card_text_by_id[cid])
                for cid in candidate_ids
            }
            cold_ranks.append(rank_of_positive(candidate_scores_cold, case.positive_card_id, random.Random(SEED + fold_idx)))
            candidate_scores_dual = {
                cid: weighted_score(case.query_text, card_text_by_id[cid], token_weights)
                for cid in candidate_ids
            }
            dual_ranks.append(rank_of_positive(candidate_scores_dual, case.positive_card_id, random.Random(SEED + fold_idx + 999)))
            for run_idx in range(RANDOM_REPEATS):
                rng = random.Random(SEED + fold_idx * 1000 + run_idx * 17 + hash(case.positive_card_id) % 997)
                random_scores = {cid: rng.random() for cid in candidate_ids}
                random_rank_runs[run_idx].append(rank_of_positive(random_scores, case.positive_card_id, rng))

    random_metrics_runs = [summarize_ranks(ranks) for ranks in random_rank_runs]
    metric_names = ["hit@1", "hit@3", "hit@5", "mrr", "mean_rank", "median_rank"]
    random_metrics = {
        name: round(sum(run[name] for run in random_metrics_runs) / len(random_metrics_runs), 4)
        for name in metric_names
    }
    random_metrics["count"] = len(random_rank_runs[0]) if random_rank_runs else 0
    random_metrics["std_mrr"] = round(statistics.pstdev(run["mrr"] for run in random_metrics_runs), 4)
    random_metrics["std_hit@1"] = round(statistics.pstdev(run["hit@1"] for run in random_metrics_runs), 4)

    cold_metrics = summarize_ranks(cold_ranks)
    dual_metrics = summarize_ranks(dual_ranks)
    cold_metrics.update(pairwise_accuracy(cold_pairwise_cases, cosine_score))
    dual_metrics.update(pairwise_accuracy(dual_pairwise_cases, lambda q, c: weighted_score(q, c, train_token_weights(cases))))
    return {
        "random": random_metrics,
        "cold_start": cold_metrics,
        "dual_tower": dual_metrics,
        "rank_distributions": {
            "random_mean_rank_per_run": [round(sum(ranks) / len(ranks), 4) for ranks in random_rank_runs],
            "cold_start": cold_ranks,
            "dual_tower": dual_ranks,
        },
    }


def evaluate_with_fold_models(cases: list[QueryCase], card_text_by_id: dict[str, str]) -> dict[str, Any]:
    scene_ids = sorted({case.scene_id for case in cases})
    folds = [scene_ids[idx::5] for idx in range(5)]
    cold_ranks: list[int] = []
    dual_raw_ranks: list[int] = []
    dual_fused_ranks: list[int] = []
    random_rank_runs: list[list[int]] = [[] for _ in range(RANDOM_REPEATS)]
    top1_by_method: dict[str, dict[str, str]] = {"random": {}, "cold_start": {}, "dual_tower_raw": {}, "dual_tower_fused": {}}
    cold_pairwise_stats = {"hard_negative_pairwise_acc": [], "hard_negative_avg_margin": [], "no_value_pairwise_acc": [], "no_value_avg_margin": []}
    dual_raw_pairwise_stats = {"hard_negative_pairwise_acc": [], "hard_negative_avg_margin": [], "no_value_pairwise_acc": [], "no_value_avg_margin": []}
    dual_fused_pairwise_stats = {"hard_negative_pairwise_acc": [], "hard_negative_avg_margin": [], "no_value_pairwise_acc": [], "no_value_avg_margin": []}
    selected_alphas: list[float] = []
    for fold_idx, test_scene_ids in enumerate(folds):
        test_scene_set = set(test_scene_ids)
        train_cases = [case for case in cases if case.scene_id not in test_scene_set]
        test_cases = [case for case in cases if case.scene_id in test_scene_set]
        token_weights = train_token_weights(train_cases)
        alpha = select_alpha(train_cases, card_text_by_id, token_weights)
        selected_alphas.append(alpha)
        fold_cold = pairwise_accuracy(test_cases, cosine_score)
        fold_dual_raw = pairwise_accuracy(test_cases, lambda q, c, tw=token_weights: weighted_score(q, c, tw))
        fold_dual_fused = pairwise_accuracy(test_cases, lambda q, c, tw=token_weights, a=alpha: fused_score(q, c, tw, a))
        for key, value in fold_cold.items():
            cold_pairwise_stats[key].append(value)
        for key, value in fold_dual_raw.items():
            dual_raw_pairwise_stats[key].append(value)
        for key, value in fold_dual_fused.items():
            dual_fused_pairwise_stats[key].append(value)
        for case in test_cases:
            candidate_ids = [case.positive_card_id] + case.negative_card_ids
            cold_scores = {cid: cosine_score(case.query_text, card_text_by_id[cid]) for cid in candidate_ids}
            dual_raw_scores = {cid: weighted_score(case.query_text, card_text_by_id[cid], token_weights) for cid in candidate_ids}
            dual_fused_scores = {cid: fused_score(case.query_text, card_text_by_id[cid], token_weights, alpha) for cid in candidate_ids}
            cold_ranks.append(rank_of_positive(cold_scores, case.positive_card_id, random.Random(SEED + fold_idx)))
            dual_raw_ranks.append(rank_of_positive(dual_raw_scores, case.positive_card_id, random.Random(SEED + fold_idx + 999)))
            dual_fused_ranks.append(rank_of_positive(dual_fused_scores, case.positive_card_id, random.Random(SEED + fold_idx + 1999)))
            top1_by_method["cold_start"][case.positive_card_id] = max(cold_scores.items(), key=lambda item: item[1])[0]
            top1_by_method["dual_tower_raw"][case.positive_card_id] = max(dual_raw_scores.items(), key=lambda item: item[1])[0]
            top1_by_method["dual_tower_fused"][case.positive_card_id] = max(dual_fused_scores.items(), key=lambda item: item[1])[0]
            for run_idx in range(RANDOM_REPEATS):
                rng = random.Random(SEED + fold_idx * 1000 + run_idx * 17 + stable_int(case.positive_card_id) % 997)
                random_scores = {cid: rng.random() for cid in candidate_ids}
                random_rank_runs[run_idx].append(rank_of_positive(random_scores, case.positive_card_id, rng))
                if run_idx == 0:
                    top1_by_method["random"][case.positive_card_id] = max(random_scores.items(), key=lambda item: item[1])[0]

    random_metrics_runs = [summarize_ranks(ranks) for ranks in random_rank_runs]
    metric_names = ["hit@1", "hit@3", "hit@5", "mrr", "mean_rank", "median_rank"]
    random_metrics = {
        name: round(sum(run[name] for run in random_metrics_runs) / len(random_metrics_runs), 4)
        for name in metric_names
    }
    random_metrics["count"] = len(random_rank_runs[0]) if random_rank_runs else 0
    random_metrics["std_mrr"] = round(statistics.pstdev(run["mrr"] for run in random_metrics_runs), 4)
    random_metrics["std_hit@1"] = round(statistics.pstdev(run["hit@1"] for run in random_metrics_runs), 4)

    cold_metrics = summarize_ranks(cold_ranks)
    dual_raw_metrics = summarize_ranks(dual_raw_ranks)
    dual_fused_metrics = summarize_ranks(dual_fused_ranks)
    for key, values in cold_pairwise_stats.items():
        cold_metrics[key] = round(sum(values) / len(values), 4)
    for key, values in dual_raw_pairwise_stats.items():
        dual_raw_metrics[key] = round(sum(values) / len(values), 4)
    for key, values in dual_fused_pairwise_stats.items():
        dual_fused_metrics[key] = round(sum(values) / len(values), 4)
    dual_fused_metrics["avg_selected_alpha"] = round(sum(selected_alphas) / len(selected_alphas), 4) if selected_alphas else 0.0

    return {
        "random": random_metrics,
        "cold_start": cold_metrics,
        "dual_tower_raw": dual_raw_metrics,
        "dual_tower_fused": dual_fused_metrics,
        "rank_distributions": {
            "random_mean_rank_per_run": [round(sum(ranks) / len(ranks), 4) for ranks in random_rank_runs],
            "cold_start": cold_ranks,
            "dual_tower_raw": dual_raw_ranks,
            "dual_tower_fused": dual_fused_ranks,
        },
        "top1_by_method": top1_by_method,
    }


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_llm_config(root: Path) -> LLMJudgeConfig | None:
    load_env(root / ".env")
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("ANTHROPIC_API_KEY", "").strip()
    model_id = os.getenv("LLM_MODEL_ID", "").strip() or os.getenv("ANTHROPIC_ENDPOINT", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if not api_key or not model_id or not base_url:
        return None
    return LLMJudgeConfig(
        api_key=api_key,
        model_id=model_id,
        base_url=base_url,
        temperature=0.0,
        max_tokens=300,
    )


def build_llm_judge_prompt(case: QueryCase, candidate_text: str) -> tuple[str, str]:
    system_prompt = (
        "你是推荐卡片评测员。只输出 JSON，不要输出任何额外文字。"
        "请判断候选卡片对当前角色在当前聊天场景下是否值得单独推送。"
    )
    user_prompt = (
        "根据下面信息评估 candidate_card。\n"
        "评分口径：\n"
        "1. role_fit: 是否匹配该角色岗位职责、画像、兴趣关键词。\n"
        "2. value_density: 是否包含任务、风险、知识、方法、复盘等高价值信息。\n"
        "3. reference_match: 与 reference_card 的核心价值是否一致。\n"
        "4. better_than_no_value: 是否明显优于无价值消息，值得单独推送。\n"
        "5. accept: 当 role_fit>=4 且 value_density>=4 且 reference_match>=4 且 better_than_no_value=true 时为 true。\n"
        "输出 JSON 字段：accept(boolean), role_fit(1-5), value_density(1-5), reference_match(1-5), better_than_no_value(boolean), reason(string<=40字)。\n\n"
        f"query_context:\n{case.query_text}\n\n"
        f"candidate_card:\n{candidate_text}\n\n"
        f"reference_card:\n{case.positive_card_text}\n\n"
        f"no_value_message:\n{case.no_value_text or '无'}\n"
    )
    return system_prompt, user_prompt


def call_llm_judge(config: LLMJudgeConfig, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    endpoint = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": 1.0,
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    text = str(data["choices"][0]["message"]["content"]).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    return {
        "accept": bool(parsed.get("accept", False)),
        "role_fit": int(parsed.get("role_fit", 1)),
        "value_density": int(parsed.get("value_density", 1)),
        "reference_match": int(parsed.get("reference_match", 1)),
        "better_than_no_value": bool(parsed.get("better_than_no_value", False)),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def evaluate_llm_judge(
    root: Path,
    cases: list[QueryCase],
    card_text_by_id: dict[str, str],
    retrieval_results: dict[str, Any],
) -> dict[str, Any]:
    if os.getenv("SOFREE_RUN_LLM_JUDGE", "0").strip() not in {"1", "true", "TRUE", "yes"}:
        return {"enabled": False, "error": "disabled"}
    config = load_llm_config(root)
    if config is None:
        return {"enabled": False, "error": "missing_llm_config"}
    ordered_cases = sorted(cases, key=lambda item: (item.scene_id, item.role_name, item.positive_card_id))
    rng = random.Random(SEED)
    sample_cases = rng.sample(ordered_cases, min(LLM_SAMPLE_CASES, len(ordered_cases)))
    top1_by_method = retrieval_results["top1_by_method"]
    per_method: dict[str, list[dict[str, Any]]] = {"random": [], "cold_start": [], "dual_tower_fused": []}
    for case in sample_cases:
        for method in per_method:
            candidate_id = top1_by_method[method][case.positive_card_id]
            candidate_text = card_text_by_id[candidate_id]
            system_prompt, user_prompt = build_llm_judge_prompt(case, candidate_text)
            judgement = call_llm_judge(config, system_prompt, user_prompt)
            per_method[method].append(
                {
                    "scene_id": case.scene_id,
                    "role_name": case.role_name,
                    "positive_card_id": case.positive_card_id,
                    "candidate_card_id": candidate_id,
                    **judgement,
                }
            )
    summary: dict[str, Any] = {
        "enabled": True,
        "sample_case_count": len(sample_cases),
        "temperature": config.temperature,
        "methods": {},
    }
    for method, rows in per_method.items():
        total = len(rows) or 1
        summary["methods"][method] = {
            "accept_rate": round(sum(1 for row in rows if row["accept"]) / total, 4),
            "better_than_no_value_rate": round(sum(1 for row in rows if row["better_than_no_value"]) / total, 4),
            "avg_role_fit": round(sum(row["role_fit"] for row in rows) / total, 4),
            "avg_value_density": round(sum(row["value_density"] for row in rows) / total, 4),
            "avg_reference_match": round(sum(row["reference_match"] for row in rows) / total, 4),
            "rows": rows,
        }
    return summary


def plot_metrics(results: dict[str, Any], output_dir: Path) -> None:
    labels = ["random", "cold_start", "dual_tower_raw", "dual_tower_fused"]
    metric_keys = ["hit@1", "hit@3", "hit@5", "mrr"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric in zip(axes.flat, metric_keys):
        values = [results[label][metric] for label in labels]
        ax.bar(labels, values, color=["#b0b0b0", "#7fb3d5", "#5dade2", "#2e86c1"])
        ax.set_title(metric)
        ax.set_ylim(0, 1 if metric != "mrr" else max(0.6, max(values) * 1.2))
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle("SoFree Offline Retrieval Evaluation")
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_bar.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    rank_data = [
        results["rank_distributions"]["cold_start"],
        results["rank_distributions"]["dual_tower_raw"],
        results["rank_distributions"]["dual_tower_fused"],
        results["rank_distributions"]["random_mean_rank_per_run"],
    ]
    ax.boxplot(
        rank_data,
        labels=["cold_start rank", "dual_raw rank", "dual_fused rank", "random mean rank/run"],
        showmeans=True,
    )
    ax.set_title("Rank Distribution Comparison")
    ax.set_ylabel("Rank")
    fig.tight_layout()
    fig.savefig(output_dir / "rank_boxplot.png", dpi=180)
    plt.close(fig)


def plot_llm_metrics(llm_results: dict[str, Any], output_dir: Path) -> None:
    if not llm_results.get("enabled"):
        return
    methods = ["random", "cold_start", "dual_tower_fused"]
    metrics = ["accept_rate", "avg_role_fit", "avg_value_density", "avg_reference_match"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric in zip(axes.flat, metrics):
        values = [llm_results["methods"][method][metric] for method in methods]
        ax.bar(methods, values, color=["#b0b0b0", "#7fb3d5", "#2e86c1"])
        ax.set_title(metric)
        ax.set_ylim(0, 1.0 if metric == "accept_rate" else 5.0)
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("LLM Judge Evaluation")
    fig.tight_layout()
    fig.savefig(output_dir / "llm_judge_bar.png", dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    workbook_path = root / "tmp" / "sofree_eval.xlsx"
    output_dir = root / "tmp" / "sofree_eval_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = parse_xlsx(workbook_path)
    cases, card_text_by_id = build_cases(data)
    results = evaluate_with_fold_models(cases, card_text_by_id)
    llm_results = evaluate_llm_judge(root, cases, card_text_by_id, results)
    metadata = {
        "case_count": len(cases),
        "scene_count": len({case.scene_id for case in cases}),
        "role_count": len({case.role_id for case in cases}),
        "candidate_card_count": len(card_text_by_id),
    }
    report = {
        "metadata": metadata,
        "results": results,
        "llm_judge": llm_results,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_metrics(results, output_dir)
    plot_llm_metrics(llm_results, output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
