from __future__ import annotations

import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SEED = 42
EMBED_DIM = 64
HIDDEN_DIM = 64
EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NEGATIVE_SAMPLE_COUNT = 4
# 损失函数margin
MARGIN = 0.0
# 融合权重: 0.65 冷启动关键词分数 + 0.35 双塔语义分数
FUSION_ALPHA = 0.65
GRADIENT_CLIP_NORM = 1.0


@dataclass
class DenseCase:
    scene_id: str
    role_id: str
    role_name: str
    query_text: str
    positive_card_id: str
    positive_card_text: str
    negative_card_ids: list[str]
    negative_card_texts: list[str]
    no_value_text: str


class DenseDualTower(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.user_embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.content_embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.user_proj = torch.nn.Linear(embed_dim, hidden_dim)
        self.content_proj = torch.nn.Linear(embed_dim, hidden_dim)

    def encode_user(self, token_ids: torch.Tensor) -> torch.Tensor:
        pooled = self._mean_pool(self.user_embedding(token_ids), token_ids)
        return F.normalize(torch.tanh(self.user_proj(pooled)), dim=-1)

    def encode_content(self, token_ids: torch.Tensor) -> torch.Tensor:
        pooled = self._mean_pool(self.content_embedding(token_ids), token_ids)
        return F.normalize(torch.tanh(self.content_proj(pooled)), dim=-1)

    def score(self, user_ids: torch.Tensor, content_ids: torch.Tensor) -> torch.Tensor:
        user_vec = self.encode_user(user_ids)
        content_vec = self.encode_content(content_ids)
        return (user_vec * content_vec).sum(dim=-1)

    @staticmethod
    def _mean_pool(embedded: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids != 0).float().unsqueeze(-1)
        summed = (embedded * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom


def load_eval_module(root: Path) -> Any:
    # 优先加载当前目录下的依赖模块
    module_path = Path(__file__).parent / "eval_sofree_dual_tower.py"
    if not module_path.exists():
        module_path = root / "tmp" / "eval_sofree_dual_tower.py"
    spec = importlib.util.spec_from_file_location("sofree_eval_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def profile_brief(text: str) -> str:
    value = str(text or "").strip()
    parts = re.split(r"[；;。.!?]", value)
    return parts[0].strip() if parts and parts[0].strip() else value[:24]


def build_compact_query(role: dict[str, str], scene: dict[str, str]) -> str:
    parts: list[str] = []
    if str(role.get("身份/岗位") or "").strip():
        parts.append(f"role_title: {role['身份/岗位']}")
    # 保留完整画像，不再只截取第一句
    profile = str(role.get("画像") or "").strip()
    if profile:
        parts.append(f"profile: {profile}")
    chat = str(scene.get("聊天片段") or "").strip()
    if chat:
        parts.append(f"chat: {chat}")
    # 增加业务域信息
    if str(scene.get("业务域") or "").strip():
        parts.append(f"domain: {scene['业务域']}")
    return " | ".join(parts)


def build_compact_cases(eval_module: Any, workbook_path: Path) -> tuple[list[DenseCase], dict[str, str]]:
    data = eval_module.parse_xlsx(workbook_path)
    roles = {row["姓名"]: row for row in data["角色库"]}
    scenes = {row["场景ID"]: row for row in data["50组聊天场景"]}
    cards = data["推送卡片明细"]
    no_value_map = {row["场景ID"]: row["不应推送的消息"] for row in data["无价值消息样本"]}
    card_text_by_id = {row["卡片ID"]: eval_module.build_card_text(row) for row in cards}

    cards_by_scene: dict[str, list[dict[str, str]]] = {}
    for card in cards:
        cards_by_scene.setdefault(card["场景ID"], []).append(card)

    cases: list[DenseCase] = []
    for card in cards:
        scene_id = card["场景ID"]
        role_name = card["推送对象"]
        role = roles[role_name]
        scene = scenes[scene_id]
        negatives: list[dict[str, str]] = []
        seen: set[str] = set()
        buckets = (
            [item for item in cards_by_scene.get(scene_id, []) if item["卡片ID"] != card["卡片ID"]],
            [item for item in cards if item["推送对象"] == role_name and item["卡片ID"] != card["卡片ID"]],
            [
                item for item in cards
                if item["业务域"] == card["业务域"]
                and item["卡片ID"] != card["卡片ID"]
                and item["推送对象"] != role_name
            ],
        )
        for bucket in buckets:
            for item in bucket:
                card_id = item["卡片ID"]
                if card_id in seen:
                    continue
                seen.add(card_id)
                negatives.append(item)
        cases.append(
            DenseCase(
                scene_id=scene_id,
                role_id=role.get("角色ID", ""),
                role_name=role_name,
                query_text=build_compact_query(role, scene),
                positive_card_id=card["卡片ID"],
                positive_card_text=card_text_by_id[card["卡片ID"]],
                negative_card_ids=[item["卡片ID"] for item in negatives],
                negative_card_texts=[card_text_by_id[item["卡片ID"]] for item in negatives],
                no_value_text=str(no_value_map.get(scene_id, "") or ""),
            )
        )
    return cases, card_text_by_id


def build_vocab(eval_module: Any, train_cases: list[DenseCase]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for case in train_cases:
        for text in [case.query_text, case.positive_card_text, *case.negative_card_texts, case.no_value_text]:
            for token in eval_module.tokenize(text):
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def encode_text(eval_module: Any, vocab: dict[str, int], text: str, max_len: int = 96) -> list[int]:
    tokens = list(eval_module.tokenize(text))[:max_len]
    if not tokens:
        return [0]
    return [vocab.get(token, 1) for token in tokens]


def pad_batch(sequences: list[list[int]]) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    rows = [seq + [0] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(rows, dtype=torch.long)


def train_dense_model(eval_module: Any, train_cases: list[DenseCase]) -> tuple[DenseDualTower, dict[str, int]]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    vocab = build_vocab(eval_module, train_cases)
    model = DenseDualTower(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        random.shuffle(train_cases)
        total_loss = 0.0
        batch_count = 0

        # 批量训练
        for i in range(0, len(train_cases), BATCH_SIZE):
            batch_cases = train_cases[i:i+BATCH_SIZE]
            user_ids_list = []
            pos_ids_list = []
            neg_ids_list = []

            for case in batch_cases:
                if not case.negative_card_texts:
                    continue

                # 选择多个负样本，优先选择和query相似度高的难负样本
                negative_scores = [(text, eval_module.cosine_score(case.query_text, text))
                                 for text in case.negative_card_texts]
                # 按相似度排序，取Top-N难负样本
                negative_scores.sort(key=lambda x: x[1], reverse=True)
                selected_negatives = [text for text, score in negative_scores[:NEGATIVE_SAMPLE_COUNT]]

                # 如果不够，用随机采样补充
                while len(selected_negatives) < NEGATIVE_SAMPLE_COUNT and case.negative_card_texts:
                    selected_negatives.append(random.choice(case.negative_card_texts))

                for neg_text in selected_negatives:
                    user_ids_list.append(encode_text(eval_module, vocab, case.query_text))
                    pos_ids_list.append(encode_text(eval_module, vocab, case.positive_card_text))
                    neg_ids_list.append(encode_text(eval_module, vocab, neg_text))

            if not user_ids_list:
                continue

            # 批量编码
            user_ids = pad_batch(user_ids_list)
            pos_ids = pad_batch(pos_ids_list)
            neg_ids = pad_batch(neg_ids_list)

            # 计算带margin的对比损失
            pos_score = model.score(user_ids, pos_ids)
            neg_score = model.score(user_ids, neg_ids)
            loss = F.softplus(-(pos_score - neg_score - MARGIN)).mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / max(batch_count, 1)
            print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return model, vocab


def score_dense(eval_module: Any, model: DenseDualTower, vocab: dict[str, int], query_text: str, content_text: str) -> float:
    model.eval()
    with torch.no_grad():
        user_ids = pad_batch([encode_text(eval_module, vocab, query_text)])
        content_ids = pad_batch([encode_text(eval_module, vocab, content_text)])
        return float(model.score(user_ids, content_ids).item())


def score_fusion(eval_module: Any, model: DenseDualTower, vocab: dict[str, int], query_text: str, content_text: str) -> float:
    """融合冷启动关键词分数和双塔语义分数"""
    cold_score = eval_module.cosine_score(query_text, content_text)
    dense_score = score_dense(eval_module, model, vocab, query_text, content_text)
    # 使用sigmoid归一化双塔分数到[0,1]区间，适配大温度系数的情况
    normalized_dense = torch.sigmoid(torch.tensor(dense_score)).item()
    return FUSION_ALPHA * cold_score + (1 - FUSION_ALPHA) * normalized_dense


def rank_of_positive(scores: dict[str, float], positive_id: str, seed: int) -> int:
    rng = random.Random(seed)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], rng.random()))
    for index, (card_id, _) in enumerate(ordered, start=1):
        if card_id == positive_id:
            return index
    return len(ordered) + 1


def summarize_ranks(ranks: list[int]) -> dict[str, float]:
    total = len(ranks)
    ndcg10 = sum((1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0 for rank in ranks) / total
    return {
        "count": total,
        "hit@1": round(sum(1 for rank in ranks if rank <= 1) / total, 4),
        "hit@3": round(sum(1 for rank in ranks if rank <= 3) / total, 4),
        "hit@5": round(sum(1 for rank in ranks if rank <= 5) / total, 4),
        "hit@10": round(sum(1 for rank in ranks if rank <= 10) / total, 4),
        "mrr": round(sum(1.0 / rank for rank in ranks) / total, 4),
        "ndcg@10": round(ndcg10, 4),
        "mean_rank": round(sum(ranks) / total, 3),
    }


def pairwise_accuracy(cases: list[DenseCase], scorer: Any) -> dict[str, float]:
    hard_win = 0
    no_value_win = 0
    hard_margin_sum = 0.0
    no_value_margin_sum = 0.0
    hard_total = 0
    no_value_total = 0
    for case in cases:
        pos_score = scorer(case.query_text, case.positive_card_text)
        if case.negative_card_texts:
            negative_scores = [scorer(case.query_text, text) for text in case.negative_card_texts]
            hard_total += 1
            margin = pos_score - max(negative_scores)
            hard_margin_sum += margin
            if margin > 0:
                hard_win += 1
        if case.no_value_text.strip():
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


def evaluate_dense_baseline(eval_module: Any, cases: list[DenseCase], card_text_by_id: dict[str, str]) -> dict[str, Any]:
    scene_ids = sorted({case.scene_id for case in cases})
    folds = [scene_ids[index::5] for index in range(5)]
    cold_ranks: list[int] = []
    dense_ranks: list[int] = []
    fusion_ranks: list[int] = []
    dense_pair_stats = {"hard_negative_pairwise_acc": [], "hard_negative_avg_margin": [], "no_value_pairwise_acc": [], "no_value_avg_margin": []}
    cold_pair_stats = {"hard_negative_pairwise_acc": [], "hard_negative_avg_margin": [], "no_value_pairwise_acc": [], "no_value_avg_margin": []}
    fusion_pair_stats = {"hard_negative_pairwise_acc": [], "hard_negative_avg_margin": [], "no_value_pairwise_acc": [], "no_value_avg_margin": []}

    for fold_index, test_scene_ids in enumerate(folds):
        print(f"\n===== Fold {fold_index + 1}/5 =====")
        test_scene_set = set(test_scene_ids)
        train_cases = [case for case in cases if case.scene_id not in test_scene_set]
        test_cases = [case for case in cases if case.scene_id in test_scene_set]
        print(f"Train cases: {len(train_cases)}, Test cases: {len(test_cases)}")

        model, vocab = train_dense_model(eval_module, train_cases)

        cold_stats = pairwise_accuracy(test_cases, lambda q, c: eval_module.cosine_score(q, c))
        dense_stats = pairwise_accuracy(test_cases, lambda q, c, mm=model, vv=vocab: score_dense(eval_module, mm, vv, q, c))
        fusion_stats = pairwise_accuracy(test_cases, lambda q, c, mm=model, vv=vocab: score_fusion(eval_module, mm, vv, q, c))

        for key, value in cold_stats.items():
            cold_pair_stats[key].append(value)
        for key, value in dense_stats.items():
            dense_pair_stats[key].append(value)
        for key, value in fusion_stats.items():
            fusion_pair_stats[key].append(value)

        for case in test_cases:
            candidate_ids = [case.positive_card_id] + case.negative_card_ids
            cold_scores = {cid: eval_module.cosine_score(case.query_text, card_text_by_id[cid]) for cid in candidate_ids}
            dense_scores = {cid: score_dense(eval_module, model, vocab, case.query_text, card_text_by_id[cid]) for cid in candidate_ids}
            fusion_scores = {cid: score_fusion(eval_module, model, vocab, case.query_text, card_text_by_id[cid]) for cid in candidate_ids}

            cold_ranks.append(rank_of_positive(cold_scores, case.positive_card_id, seed=fold_index + 1))
            dense_ranks.append(rank_of_positive(dense_scores, case.positive_card_id, seed=fold_index + 101))
            fusion_ranks.append(rank_of_positive(fusion_scores, case.positive_card_id, seed=fold_index + 201))

    cold_metrics = summarize_ranks(cold_ranks)
    dense_metrics = summarize_ranks(dense_ranks)
    fusion_metrics = summarize_ranks(fusion_ranks)

    for key, values in cold_pair_stats.items():
        cold_metrics[key] = round(sum(values) / len(values), 4)
    for key, values in dense_pair_stats.items():
        dense_metrics[key] = round(sum(values) / len(values), 4)
    for key, values in fusion_pair_stats.items():
        fusion_metrics[key] = round(sum(values) / len(values), 4)

    print("\n===== 效果对比 =====")
    print(f"冷启动 Hit@1: {cold_metrics['hit@1']:.4f}, Hit@3: {cold_metrics['hit@3']:.4f}, MRR: {cold_metrics['mrr']:.4f}")
    print(f"双塔模型 Hit@1: {dense_metrics['hit@1']:.4f}, Hit@3: {dense_metrics['hit@3']:.4f}, MRR: {dense_metrics['mrr']:.4f}")
    print(f"融合模型 Hit@1: {fusion_metrics['hit@1']:.4f}, Hit@3: {fusion_metrics['hit@3']:.4f}, MRR: {fusion_metrics['mrr']:.4f}")

    return {
        "cold_start": cold_metrics,
        "dense_dual_tower": dense_metrics,
        "fusion_model": fusion_metrics
    }


def main() -> None:
    # 适配新的目录结构
    current_dir = Path(__file__).parent
    # 优先读取当前目录下的测试集
    workbook_path = current_dir / "SoFree_50组角色画像价值消息测试集.xlsx"
    if not workbook_path.exists():
        workbook_path = Path(r"D:\下载\SoFree_50组角色画像价值消息测试集.xlsx")

    eval_module = load_eval_module(Path(__file__).resolve().parents[3])
    cases, card_text_by_id = build_compact_cases(eval_module, workbook_path)
    results = evaluate_dense_baseline(eval_module, cases, card_text_by_id)

    report = {
        "metadata": {
            "workbook": str(workbook_path),
            "case_count": len(cases),
            "scene_count": len({case.scene_id for case in cases}),
            "query_mode": "compact_role_chat",
            "model": {
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "epochs": EPOCHS,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        },
        "results": results,
    }

    # 输出到当前目录下的outputs文件夹
    output_dir = current_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "dense_dual_tower_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n✅ 评估完成，报告已保存到: {output_path.resolve()}")


if __name__ == "__main__":
    main()
