import os
import re
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

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
    token_indices: List[int]  # Global BERT token indices (without special tokens)


class JiebaNgramSegmenter:
    # Keep mixed-language and domain terms as a single protected span.
    DEFAULT_PROTECTED_PATTERNS = [
        re.compile(r"https?://[^\s]+", re.IGNORECASE),
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_/+.][A-Za-z0-9]+)+"),  # e.g. A/B, GPT-4o, anti-fraud_v2
        re.compile(r"[A-Za-z]+[A-Za-z0-9]*"),  # e.g. CRM, OpenAI, L4
        re.compile(r"\d+(?:\.\d+)?(?:%|ms|s|m|h|Hz|kHz|MHz|GHz|GB|MB|TB)?"),
    ]
    SYMBOL_MAP = {
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "—": "-",
        "－": "-",
        "～": "~",
    }

    def __init__(
        self,
        user_dict_path: Optional[str] = None,
        custom_terms: Optional[List[str]] = None,
        min_count: int = 2,
        min_pmi: float = 4.0,
        max_n: int = 3,
    ):
        try:
            import jieba  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "jieba is required. Please install it with `pip install jieba`."
            ) from exc
        self._jieba = jieba
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.max_n = max_n
        self._dynamic_terms: Set[str] = set()
        self._custom_terms = [term.strip() for term in (custom_terms or []) if term and term.strip()]
        self._normalized_custom_terms = [self._normalize_text(term) for term in self._custom_terms]
        if user_dict_path and os.path.exists(user_dict_path):
            self._jieba.load_userdict(user_dict_path)
        for term in self._custom_terms:
            self._jieba.add_word(term)

    def segment_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        if not text:
            return []

        normalized_text = self._normalize_text(text)
        protected = self._collect_non_overlapping_spans(normalized_text)
        segments: List[Tuple[str, int, int]] = []

        cursor = 0
        for start, end in protected:
            if cursor < start:
                segments.extend(self._segment_plain_text(text, normalized_text, cursor, start))
            protected_word = text[start:end]
            if protected_word.strip():
                segments.append((protected_word, start, end))
            cursor = end

        if cursor < len(text):
            segments.extend(self._segment_plain_text(text, normalized_text, cursor, len(text)))

        return [(w, s, e) for (w, s, e) in segments if w and w.strip()]

    def _segment_plain_text(
        self,
        original_text: str,
        normalized_text: str,
        start: int,
        end: int,
    ) -> List[Tuple[str, int, int]]:
        piece = normalized_text[start:end]
        if not piece.strip():
            return []

        results: List[Tuple[str, int, int]] = []
        token_spans = [(token, t_start, t_end) for token, t_start, t_end in self._jieba.tokenize(piece, mode="default") if token and token.strip()]
        if not token_spans:
            return []

        token_texts = [token for token, _, _ in token_spans]
        candidates = self._mine_ngram_candidates(token_texts)

        dynamic_terms = {"".join(item) for item in candidates["2"]}
        if self.max_n >= 3:
            dynamic_terms.update({"".join(item) for item in candidates["3"]})
        for term in dynamic_terms:
            if term and term not in self._dynamic_terms:
                self._jieba.add_word(term)
                self._dynamic_terms.add(term)

        # Re-tokenize after dynamic lexicon injection, then do greedy n-gram merge.
        token_spans = [(token, t_start, t_end) for token, t_start, t_end in self._jieba.tokenize(piece, mode="default") if token and token.strip()]
        token_texts = [token for token, _, _ in token_spans]
        candidates = self._mine_ngram_candidates(token_texts)

        merged_spans = self._merge_by_ngram(token_spans, candidates)
        for _, local_start, local_end in merged_spans:
            global_start = start + local_start
            global_end = start + local_end
            word = original_text[global_start:global_end]
            if word.strip():
                results.append((word, global_start, global_end))
        return results

    def _mine_ngram_candidates(self, tokens: List[str]) -> Dict[str, Set[Tuple[str, ...]]]:
        candidates: Dict[str, Set[Tuple[str, ...]]] = {
            "2": set(),
            "3": set(),
        }
        if len(tokens) < 2:
            return candidates

        unigram = Counter(tokens)
        bigram = Counter(zip(tokens[:-1], tokens[1:]))
        total_unigram = max(1, sum(unigram.values()))

        for (w1, w2), freq in bigram.items():
            if freq < self.min_count:
                continue
            denom = unigram[w1] * unigram[w2]
            if denom <= 0:
                continue
            pmi = math.log2((freq * total_unigram) / denom)
            if pmi >= self.min_pmi:
                candidates["2"].add((w1, w2))

        if self.max_n >= 3 and len(tokens) >= 3:
            trigram = Counter(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
            for (w1, w2, w3), freq in trigram.items():
                if freq < self.min_count:
                    continue
                if (w1, w2) in candidates["2"] and (w2, w3) in candidates["2"]:
                    candidates["3"].add((w1, w2, w3))

        return candidates

    def _merge_by_ngram(
        self,
        token_spans: List[Tuple[str, int, int]],
        candidates: Dict[str, Set[Tuple[str, ...]]],
    ) -> List[Tuple[str, int, int]]:
        merged: List[Tuple[str, int, int]] = []
        i = 0
        n = len(token_spans)
        while i < n:
            if self.max_n >= 3 and i + 2 < n:
                tri = (token_spans[i][0], token_spans[i + 1][0], token_spans[i + 2][0])
                if tri in candidates["3"]:
                    merged.append(("".join(tri), token_spans[i][1], token_spans[i + 2][2]))
                    i += 3
                    continue

            if i + 1 < n:
                bi = (token_spans[i][0], token_spans[i + 1][0])
                if bi in candidates["2"]:
                    merged.append(("".join(bi), token_spans[i][1], token_spans[i + 1][2]))
                    i += 2
                    continue

            merged.append(token_spans[i])
            i += 1

        return merged

    def _collect_non_overlapping_spans(self, normalized_text: str) -> List[Tuple[int, int]]:
        candidates: List[Tuple[int, int]] = []
        for term in self._normalized_custom_terms:
            start = 0
            while True:
                idx = normalized_text.find(term, start)
                if idx < 0:
                    break
                candidates.append((idx, idx + len(term)))
                start = idx + len(term)

        for pattern in self.DEFAULT_PROTECTED_PATTERNS:
            for match in pattern.finditer(normalized_text):
                start, end = match.span()
                if end > start:
                    candidates.append((start, end))

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        merged: List[Tuple[int, int]] = []
        current_end = -1
        for start, end in candidates:
            if start < current_end:
                continue
            merged.append((start, end))
            current_end = end
        return merged

    def _normalize_text(self, text: str) -> str:
        normalized_chars: List[str] = []
        for ch in text:
            normalized_chars.append(self._normalize_char(ch))
        return "".join(normalized_chars).lower()

    def _normalize_char(self, ch: str) -> str:
        code = ord(ch)
        if code == 0x3000:
            return " "
        if 0xFF01 <= code <= 0xFF5E:
            return chr(code - 0xFEE0)
        return self.SYMBOL_MAP.get(ch, ch)

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
        self.segmenter = JiebaNgramSegmenter(
            user_dict_path=user_dict_path,
            custom_terms=custom_terms,
            min_count=ngram_min_count,
            min_pmi=ngram_min_pmi,
            max_n=ngram_max_n,
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

        words = self.segmenter.segment_with_spans(text)
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
                # Keep tokens that overlap this word span.
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

                # Avoid duplicated evaluation in overlap area.
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
