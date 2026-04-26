from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from token_classify.domain_tokenizer import DomainAdaptiveTokenizer
from token_classify.model_loader import HFModelLoader


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def as_model_inputs(self) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


@dataclass
class WordInstance:
    word: str
    char_start: int
    char_end: int
    token_indices: List[int]


class SemanticDensityAnalyzer:
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        chunk_size: int = 256,
        overlap_ratio: float = 0.2,
        user_dict_path: Optional[str] = None,
        custom_terms: Optional[List[str]] = None,
        ngram_min_count: int = 2,
        ngram_min_pmi: float = 4.0,
        ngram_max_n: int = 3,
    ):
        self.segmenter = DomainAdaptiveTokenizer(
            user_dict_path=user_dict_path,
            custom_terms=custom_terms,
            ngram_min_count=ngram_min_count,
            ngram_min_pmi=ngram_min_pmi,
            ngram_max_n=ngram_max_n,
        )

        self.loader = HFModelLoader(model_name=model_name, device=device, hf_token=hf_token)
        self.tokenizer, self.model = self.loader.load()
        self.static_embeddings = self.model.embeddings.word_embeddings

        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio

        self._cached_text: Optional[str] = None
        self._cached_scores: Optional[List[Dict[str, object]]] = None

    def semantic_density(self, text: str) -> Tuple[List[str], List[float]]:
        scores = self._collect_word_metrics(text)
        if not scores:
            return [], []
        words = [str(score["word"]) for score in scores]
        values = [float(score["semantic_density"]) for score in scores]
        return words, values

    def attention_entropy(self, text: str) -> Tuple[List[str], List[float]]:
        scores = self._collect_word_metrics(text)
        if not scores:
            return [], []
        words = [str(score["word"]) for score in scores]
        values = [float(score["attention_entropy"]) for score in scores]
        return words, values

    def analyze_meaningful_tokens(
        self,
        text: str,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        threshold_mode: str = "median",
    ) -> Dict[str, object]:
        scores = self._collect_word_metrics(text)
        ordered_tokens = [str(item["word"]) for item in scores]
        if not ordered_tokens:
            return {
                "tokens_in_order": [],
                "meaningful_tokens": [],
                "meaningless_tokens": [],
                "meaningful_tokens_in_order": [],
                "threshold": 0.0,
                "token_scores": {},
            }

        vocabulary: List[str] = []
        seen = set()
        for token in ordered_tokens:
            if token in seen:
                continue
            seen.add(token)
            vocabulary.append(token)

        agg: Dict[str, Dict[str, float]] = {}
        for item in scores:
            token = str(item["word"])
            rec = agg.setdefault(token, {"count": 0.0, "sum_sd": 0.0, "sum_ae": 0.0})
            rec["count"] += 1.0
            rec["sum_sd"] += float(item["semantic_density"])
            rec["sum_ae"] += float(item["attention_entropy"])

        sd_values: List[float] = []
        ae_values: List[float] = []
        metric_by_token: Dict[str, Dict[str, float]] = {}
        for token in vocabulary:
            rec = agg.get(token, {"count": 0.0, "sum_sd": 0.0, "sum_ae": 0.0})
            count = rec["count"] or 1.0
            sd = rec["sum_sd"] / count
            ae = rec["sum_ae"] / count
            sd_values.append(sd)
            ae_values.append(ae)
            metric_by_token[token] = {"semantic_density": sd, "attention_entropy": ae}

        sd_norm = self._minmax(sd_values)
        ae_norm = self._minmax(ae_values)
        scores_vec = [alpha * sd - beta * ae for sd, ae in zip(sd_norm, ae_norm)]

        if threshold_mode != "median":
            raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
        threshold = median(scores_vec) if scores_vec else 0.0

        meaningful_set = set()
        meaningless_set = set()
        token_scores: Dict[str, Dict[str, float]] = {}
        for token, score in zip(vocabulary, scores_vec):
            sd = metric_by_token[token]["semantic_density"]
            ae = metric_by_token[token]["attention_entropy"]
            token_scores[token] = {
                "semantic_density": sd,
                "attention_entropy": ae,
                "score": score,
            }
            if score >= threshold:
                meaningful_set.add(token)
            else:
                meaningless_set.add(token)

        meaningful_tokens_in_order = [token for token in ordered_tokens if token in meaningful_set]
        return {
            "tokens_in_order": ordered_tokens,
            "meaningful_tokens": sorted(meaningful_set),
            "meaningless_tokens": sorted(meaningless_set),
            "meaningful_tokens_in_order": meaningful_tokens_in_order,
            "threshold": float(threshold),
            "token_scores": token_scores,
        }

    def _collect_word_metrics(self, text: str) -> List[Dict[str, object]]:
        if self._cached_text == text and self._cached_scores is not None:
            return self._cached_scores

        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        if not input_ids:
            self._cached_text = text
            self._cached_scores = []
            return []

        words = self.segmenter.tokenize_with_spans(text)
        instances = self._build_word_instances(words, offsets)
        scores = self._compute_metrics_for_instances(input_ids, instances)

        self._cached_text = text
        self._cached_scores = scores
        return scores

    def _build_word_instances(
        self,
        segmented_words: List[Tuple[str, int, int]],
        token_offsets: Sequence[Tuple[int, int]],
    ) -> List[WordInstance]:
        instances: List[WordInstance] = []
        for word, char_start, char_end in segmented_words:
            token_indices: List[int] = []
            for idx, (tok_start, tok_end) in enumerate(token_offsets):
                if tok_end <= char_start or tok_start >= char_end:
                    continue
                token_indices.append(idx)

            if token_indices:
                instances.append(
                    WordInstance(
                        word=word,
                        char_start=char_start,
                        char_end=char_end,
                        token_indices=token_indices,
                    )
                )
        return instances

    def _compute_metrics_for_instances(
        self,
        input_ids: Sequence[int],
        instances: List[WordInstance],
    ) -> List[Dict[str, object]]:
        if not instances:
            return []

        step = max(1, int(self.chunk_size * (1 - self.overlap_ratio)))
        seen_instances: Set[Tuple[int, int, str]] = set()
        all_scores: List[Dict[str, object]] = []

        for chunk_ids, chunk_start in self._iter_chunks(input_ids):
            chunk_end = chunk_start + len(chunk_ids)
            batch = self._build_batch(chunk_ids)
            outputs = self._forward(batch)
            context_emb = outputs.last_hidden_state[0]
            static_emb = self._static_word_embeddings(batch.input_ids)
            layers = list(range(max(0, len(outputs.attentions) - 4), len(outputs.attentions)))

            for inst in instances:
                key = (inst.char_start, inst.char_end, inst.word)
                if key in seen_instances:
                    continue
                if inst.token_indices[0] < chunk_start or inst.token_indices[-1] >= chunk_end:
                    continue

                first_tok = inst.token_indices[0]
                if first_tok >= chunk_start + step and chunk_end < len(input_ids):
                    continue

                local_indices = [idx - chunk_start + 1 for idx in inst.token_indices]
                semantic = self._compute_semantic_density(context_emb, static_emb, local_indices)
                attn_entropy = self._compute_attention_entropy(outputs.attentions, local_indices, layers)

                all_scores.append(
                    {
                        "word": inst.word,
                        "semantic_density": semantic,
                        "attention_entropy": attn_entropy,
                    }
                )
                seen_instances.add(key)

        return all_scores

    def _iter_chunks(self, input_ids: Sequence[int]):
        step = max(1, int(self.chunk_size * (1 - self.overlap_ratio)))
        for start in range(0, len(input_ids), step):
            end = min(start + self.chunk_size, len(input_ids))
            yield input_ids[start:end], start
            if end >= len(input_ids):
                break

    def _build_batch(self, chunk_ids: Sequence[int]) -> EncodedBatch:
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        model_input_ids = [cls_id] + list(chunk_ids) + [sep_id]
        input_ids = torch.tensor([model_input_ids], dtype=torch.long, device=self.loader.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.loader.device)
        return EncodedBatch(input_ids=input_ids, attention_mask=attention_mask)

    def _forward(self, batch: EncodedBatch):
        with torch.no_grad():
            return self.model(**batch.as_model_inputs())

    def _static_word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            static_emb = self.static_embeddings(input_ids.to(self.loader.device))
        return static_emb[0]

    @staticmethod
    def _compute_semantic_density(
        context_emb: torch.Tensor,
        static_emb: torch.Tensor,
        token_indices: List[int],
    ) -> float:
        valid_indices = [idx for idx in token_indices if idx < len(context_emb) and idx < len(static_emb)]
        if not valid_indices:
            return 0.0

        index_tensor = torch.tensor(valid_indices, device=context_emb.device, dtype=torch.long)
        ctx_vecs = torch.index_select(context_emb, dim=0, index=index_tensor)
        stc_vecs = torch.index_select(static_emb, dim=0, index=index_tensor)
        similarities = F.cosine_similarity(ctx_vecs, stc_vecs, dim=-1)
        return float(similarities.mean().item())

    @staticmethod
    def _compute_attention_entropy(
        attentions: Tuple[torch.Tensor, ...],
        token_indices: List[int],
        layers: List[int],
    ) -> float:
        if not token_indices:
            return 0.0

        entropy_values: List[torch.Tensor] = []
        for layer_idx in layers:
            layer_attention = attentions[layer_idx][0]
            for head_idx in range(layer_attention.shape[0]):
                head_attention = layer_attention[head_idx]
                for token_idx in token_indices:
                    if token_idx >= head_attention.shape[0]:
                        continue
                    attention_weights = head_attention[token_idx]
                    valid_weights = attention_weights[attention_weights > 1e-10]
                    if valid_weights.numel() <= 1:
                        continue
                    normalized = valid_weights / valid_weights.sum()
                    ent = -(normalized * torch.log(normalized)).sum()
                    max_entropy = torch.log(
                        torch.tensor(
                            float(valid_weights.numel()),
                            device=valid_weights.device,
                            dtype=valid_weights.dtype,
                        )
                    )
                    if max_entropy.item() > 0:
                        entropy_values.append(ent / max_entropy)

        if not entropy_values:
            return 0.0
        return float(torch.stack(entropy_values).mean().item())

    @staticmethod
    def _minmax(values: List[float]) -> List[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi - lo < 1e-12:
            return [0.5 for _ in values]
        return [(val - lo) / (hi - lo) for val in values]
