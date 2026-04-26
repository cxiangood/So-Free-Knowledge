#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context extraction helpers for keyword instances and overlap grouping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from token_classify.domain_tokenizer import DomainAdaptiveTokenizer

DEFAULT_MESSAGE_WINDOW_SENTENCES = 4
DEFAULT_TEXT_WINDOW_TOKENS = 80

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])")
_FALLBACK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_.+-]+")


@dataclass(slots=True)
class ContextInstance:
    instance_id: str
    keyword: str
    context: str
    source_type: str
    related_keywords: List[str]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _normalize_keywords(keywords: Sequence[str]) -> List[str]:
    return [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]


def _split_sentences(text: str) -> List[str]:
    pieces = _SENTENCE_SPLIT_RE.split(text.replace("\r\n", "\n"))
    sentences = [piece.strip() for piece in pieces if piece and piece.strip()]
    if not sentences and text.strip():
        return [text.strip()]
    return sentences


def _contains_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def _related_keywords(context: str, keywords: Sequence[str]) -> List[str]:
    matches = [kw for kw in keywords if _contains_keyword(context, kw)]
    return sorted(set(matches))


def extract_keyword_context_instances_from_messages(
    messages: Sequence[dict],
    keywords: Sequence[str],
    *,
    window_sentences: int = DEFAULT_MESSAGE_WINDOW_SENTENCES,
    content_key: str = "content",
) -> List[Dict[str, object]]:
    """Extract instance-level contexts from message collections."""
    clean_keywords = _normalize_keywords(keywords)
    if not messages or not clean_keywords:
        return []

    sentences: List[str] = []
    for msg in messages:
        content = msg.get(content_key, "") if isinstance(msg, dict) else ""
        if isinstance(content, str) and content.strip():
            sentences.extend(_split_sentences(content))

    instances: List[ContextInstance] = []
    for idx, sentence in enumerate(sentences):
        for kw in clean_keywords:
            if not _contains_keyword(sentence, kw):
                continue
            start = max(0, idx - window_sentences)
            end = min(len(sentences), idx + window_sentences + 1)
            context = "\n".join(sentences[start:end]).strip()
            related = _related_keywords(context, clean_keywords)
            instances.append(
                ContextInstance(
                    instance_id=f"msg-{len(instances)}",
                    keyword=kw,
                    context=context,
                    source_type="messages",
                    related_keywords=related,
                )
            )

    return [
        {
            "instance_id": item.instance_id,
            "keyword": item.keyword,
            "context": item.context,
            "source_type": item.source_type,
            "related_keywords": item.related_keywords,
        }
        for item in instances
    ]


def extract_keyword_context_instances_from_text(
    text: str,
    keywords: Sequence[str],
    *,
    window_tokens: int = DEFAULT_TEXT_WINDOW_TOKENS,
    tokenizer: DomainAdaptiveTokenizer | None = None,
) -> List[Dict[str, object]]:
    """Extract instance-level contexts from plain text by token window."""
    clean_keywords = _normalize_keywords(keywords)
    if not text or not clean_keywords:
        return []

    if tokenizer is None:
        try:
            tokenizer = DomainAdaptiveTokenizer()
        except Exception:
            tokenizer = None

    if tokenizer is None:
        tokens = _FALLBACK_TOKEN_RE.findall(text)
    else:
        tokens = tokenizer.tokenize(text)
    if not tokens:
        return []

    lowered = [tok.lower() for tok in tokens]
    instances: List[ContextInstance] = []
    for idx, tok in enumerate(tokens):
        tok_lower = lowered[idx]
        for kw in clean_keywords:
            if tok_lower != kw.lower():
                continue
            start = max(0, idx - window_tokens)
            end = min(len(tokens), idx + window_tokens + 1)
            context_tokens = tokens[start:end]
            context = " ".join(context_tokens).strip()
            related = _related_keywords(context, clean_keywords)
            instances.append(
                ContextInstance(
                    instance_id=f"txt-{len(instances)}",
                    keyword=kw,
                    context=context,
                    source_type="text",
                    related_keywords=related,
                )
            )

    return [
        {
            "instance_id": item.instance_id,
            "keyword": item.keyword,
            "context": item.context,
            "source_type": item.source_type,
            "related_keywords": item.related_keywords,
        }
        for item in instances
    ]


def group_context_instances(instances: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Group context instances when keywords mutually appear in each other's contexts."""
    if not instances:
        return []

    uf = _UnionFind(len(instances))
    by_keyword: Dict[str, List[int]] = {}
    for idx, inst in enumerate(instances):
        kw = str(inst.get("keyword", ""))
        by_keyword.setdefault(kw, []).append(idx)

    for idx, inst in enumerate(instances):
        keyword = str(inst.get("keyword", ""))
        related = {str(x) for x in inst.get("related_keywords", []) if str(x)}
        for other_kw in related:
            if other_kw == keyword:
                continue
            for j in by_keyword.get(other_kw, []):
                other_inst = instances[j]
                other_related = {str(x) for x in other_inst.get("related_keywords", []) if str(x)}
                if keyword in other_related:
                    uf.union(idx, j)

    grouped: Dict[int, List[int]] = {}
    for idx in range(len(instances)):
        root = uf.find(idx)
        grouped.setdefault(root, []).append(idx)

    groups: List[Dict[str, object]] = []
    for i, member_ids in enumerate(grouped.values()):
        keywords = sorted({str(instances[m].get("keyword", "")) for m in member_ids if str(instances[m].get("keyword", ""))})
        contexts: List[str] = []
        seen = set()
        for m in member_ids:
            ctx = str(instances[m].get("context", "")).strip()
            if ctx and ctx not in seen:
                contexts.append(ctx)
                seen.add(ctx)

        groups.append(
            {
                "group_id": f"group-{i}",
                "keywords": keywords,
                "contexts": contexts,
                "instance_ids": [str(instances[m].get("instance_id", "")) for m in member_ids],
            }
        )

    return groups


# Backward-compatible wrappers

def extract_contexts(
    text: str,
    keywords: List[str],
    window_size: int = DEFAULT_TEXT_WINDOW_TOKENS,
    merge_overlap: bool = True,
) -> List[str]:
    instances = extract_keyword_context_instances_from_text(text, keywords, window_tokens=window_size)
    contexts = [str(item["context"]) for item in instances]
    if not merge_overlap:
        return contexts

    deduped: List[str] = []
    seen = set()
    for ctx in contexts:
        if ctx not in seen:
            deduped.append(ctx)
            seen.add(ctx)
    return deduped


def extract_contexts_from_messages(
    messages: List[dict],
    keywords: List[str],
    window_size: int = DEFAULT_MESSAGE_WINDOW_SENTENCES,
    content_key: str = "content",
) -> List[str]:
    instances = extract_keyword_context_instances_from_messages(
        messages,
        keywords,
        window_sentences=window_size,
        content_key=content_key,
    )
    return [str(item["context"]) for item in instances]

