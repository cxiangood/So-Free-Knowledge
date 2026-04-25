import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


class DomainAdaptiveTokenizer:
    DEFAULT_PROTECTED_PATTERNS = [
        re.compile(r"https?://[^\s]+", re.IGNORECASE),
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_/+.][A-Za-z0-9]+)+"),
        re.compile(r"[A-Za-z]+[A-Za-z0-9]*"),
        re.compile(r"\d+(?:\.\d+)?(?:%|ms|s|m|h|hz|khz|mhz|ghz|gb|mb|tb)?", re.IGNORECASE),
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
        ngram_min_count: int = 2,
        ngram_min_pmi: float = 4.0,
        ngram_max_n: int = 3,
    ):
        try:
            import jieba  # type: ignore
        except ImportError as exc:
            raise ImportError("jieba is required. Please install it with `pip install jieba`.") from exc

        self._jieba = jieba
        self.ngram_min_count = ngram_min_count
        self.ngram_min_pmi = ngram_min_pmi
        self.ngram_max_n = ngram_max_n

        self._custom_terms = [term.strip() for term in (custom_terms or []) if term and term.strip()]
        self._normalized_custom_terms = [self._normalize_text(term) for term in self._custom_terms]

        if user_dict_path and os.path.exists(user_dict_path):
            self._jieba.load_userdict(user_dict_path)
        for term in self._custom_terms:
            self._jieba.add_word(term)

    def tokenize(self, text: str) -> List[str]:
        return [token for token, _, _ in self.tokenize_with_spans(text)]

    def tokenize_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        if not text:
            return []

        # Required order: normalize first, then jieba tokenization and n-gram merging.
        normalized_text = self._normalize_text(text)
        protected = self._collect_non_overlapping_spans(normalized_text)
        segments: List[Tuple[str, int, int]] = []

        cursor = 0
        for start, end in protected:
            if cursor < start:
                segments.extend(self._segment_plain_text(normalized_text, cursor, start))
            token = normalized_text[start:end]
            if token.strip():
                segments.append((token, start, end))
            cursor = end

        if cursor < len(normalized_text):
            segments.extend(self._segment_plain_text(normalized_text, cursor, len(normalized_text)))

        return [(token, start, end) for token, start, end in segments if token and token.strip()]

    def _segment_plain_text(
        self,
        normalized_text: str,
        start: int,
        end: int,
    ) -> List[Tuple[str, int, int]]:
        piece = normalized_text[start:end]
        if not piece.strip():
            return []

        token_spans = [
            (token, t_start, t_end)
            for token, t_start, t_end in self._jieba.tokenize(piece, mode="default")
            if token and token.strip()
        ]
        if not token_spans:
            return []

        token_texts = [token for token, _, _ in token_spans]
        candidates = self._mine_ngram_candidates(token_texts)

        dynamic_terms = {"".join(item) for item in candidates["2"]}
        if self.ngram_max_n >= 3:
            dynamic_terms.update({"".join(item) for item in candidates["3"]})
        for term in dynamic_terms:
            if term:
                self._jieba.add_word(term)

        token_spans = [
            (token, t_start, t_end)
            for token, t_start, t_end in self._jieba.tokenize(piece, mode="default")
            if token and token.strip()
        ]
        token_texts = [token for token, _, _ in token_spans]
        candidates = self._mine_ngram_candidates(token_texts)
        merged_spans = self._merge_by_ngram(token_spans, candidates)

        results: List[Tuple[str, int, int]] = []
        for _, local_start, local_end in merged_spans:
            g_start = start + local_start
            g_end = start + local_end
            token = normalized_text[g_start:g_end]
            if token.strip():
                results.append((token, g_start, g_end))
        return results

    def _mine_ngram_candidates(self, tokens: List[str]) -> Dict[str, Set[Tuple[str, ...]]]:
        candidates: Dict[str, Set[Tuple[str, ...]]] = {"2": set(), "3": set()}
        if len(tokens) < 2:
            return candidates

        unigram = Counter(tokens)
        bigram = Counter(zip(tokens[:-1], tokens[1:]))
        total_unigram = max(1, sum(unigram.values()))

        for (w1, w2), freq in bigram.items():
            if freq < self.ngram_min_count:
                continue
            denom = unigram[w1] * unigram[w2]
            if denom <= 0:
                continue
            pmi = math.log2((freq * total_unigram) / denom)
            if pmi >= self.ngram_min_pmi:
                candidates["2"].add((w1, w2))

        if self.ngram_max_n >= 3 and len(tokens) >= 3:
            trigram = Counter(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
            for (w1, w2, w3), freq in trigram.items():
                if freq < self.ngram_min_count:
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
            if self.ngram_max_n >= 3 and i + 2 < n:
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
        return "".join(self._normalize_char(ch) for ch in text)

    def _normalize_char(self, ch: str) -> str:
        code = ord(ch)
        if code == 0x3000:
            return " "
        if 0xFF01 <= code <= 0xFF5E:
            return chr(code - 0xFEE0)
        return self.SYMBOL_MAP.get(ch, ch)


def tokenize_text(
    text: str,
    user_dict_path: Optional[str] = None,
    custom_terms: Optional[List[str]] = None,
    ngram_min_count: int = 2,
    ngram_min_pmi: float = 4.0,
    ngram_max_n: int = 3,
) -> List[str]:
    tokenizer = DomainAdaptiveTokenizer(
        user_dict_path=user_dict_path,
        custom_terms=custom_terms,
        ngram_min_count=ngram_min_count,
        ngram_min_pmi=ngram_min_pmi,
        ngram_max_n=ngram_max_n,
    )
    return tokenizer.tokenize(text)
