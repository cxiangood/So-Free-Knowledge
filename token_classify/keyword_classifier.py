#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyword classification powered by local llm package."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from llm.client import LLMClient, LLMConfig

VALID_TYPES = {"key", "black", "confused", "nothing"}


class KeywordClassifier:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ):
        config = LLMConfig.from_env(
            api_key=api_key or "",
            model_id=model_id or "",
            base_url=base_url or "",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.client = LLMClient(config)
        self.system_prompt = self._build_system_prompt()

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "你是关键词分类助手。请对一组关键词分别判断类型与含义。"
            "类型只能是 key/black/confused/nothing。"
            "输出严格 JSON，不要输出任何额外文本。"
            'JSON 格式: {"items":[{"keyword":"...","type":"key","sense":"..."}]}。'
            "当 type 为 nothing 时，sense 必须为空字符串。"
        )

    @staticmethod
    def _default_result(keywords: List[str], reason: str) -> Dict[str, Dict[str, str]]:
        return {kw: {"type": "confused", "sense": reason} for kw in keywords}

    @staticmethod
    def _extract_json_block(text: str) -> Optional[dict]:
        text = text.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
        return None

    @staticmethod
    def _normalize_items(payload: dict, keywords: List[str]) -> Dict[str, Dict[str, str]]:
        out = {kw: {"type": "confused", "sense": "模型未返回该关键词结果"} for kw in keywords}

        items = payload.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                keyword = item.get("keyword")
                if keyword not in out:
                    continue
                item_type = str(item.get("type", "confused"))
                if item_type not in VALID_TYPES:
                    item_type = "confused"
                sense = str(item.get("sense", ""))
                if item_type == "nothing":
                    sense = ""
                out[keyword] = {"type": item_type, "sense": sense}
            return out

        # Compatibility fallback: {"keyword":{"type":"...","sense":"..."}}
        for keyword in keywords:
            raw = payload.get(keyword)
            if not isinstance(raw, dict):
                continue
            item_type = str(raw.get("type", "confused"))
            if item_type not in VALID_TYPES:
                item_type = "confused"
            sense = str(raw.get("sense", raw.get("value", "")))
            if item_type == "nothing":
                sense = ""
            out[keyword] = {"type": item_type, "sense": sense}
        return out

    def classify_group_keywords(self, keywords: List[str], contexts: List[str]) -> Dict[str, Dict[str, str]]:
        if not keywords:
            return {}

        context_lines = "\n".join(f"- {ctx}" for ctx in contexts[:8]) if contexts else "- 无上下文"
        user_prompt = (
            "请按组处理以下关键词，并分别判断每个关键词：\n"
            f"关键词组: {', '.join(keywords)}\n"
            "上下文片段（这些片段可能同时涉及多个关键词）:\n"
            f"{context_lines}\n\n"
            "请输出 JSON。"
        )

        print("\n=== LLM INPUT: system_prompt ===")
        print(self.system_prompt)
        print("\n=== LLM INPUT: user_prompt ===")
        print(user_prompt)

        response_text = self.client.build_reply(self.system_prompt, user_prompt)
        print("\n=== LLM OUTPUT: response_text ===")
        print(response_text)
        if not isinstance(response_text, str):
            return self._default_result(keywords, "模型返回异常")

        if response_text.startswith("LLM "):
            return self._default_result(keywords, response_text)

        payload = self._extract_json_block(response_text)
        if not payload:
            return self._default_result(keywords, f"模型输出非JSON: {response_text[:120]}")

        return self._normalize_items(payload, keywords)

    def batch_classify_with_groups(self, groups: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Returns:
            {
              "group-0": {"关键词A":{"type":"...","sense":"..."}, ...},
              ...
            }
        """
        out: Dict[str, Dict[str, Dict[str, str]]] = {}
        for group in groups:
            group_id = str(group.get("group_id", ""))
            keywords = [str(x) for x in group.get("keywords", []) if str(x)]
            contexts = [str(x) for x in group.get("contexts", []) if str(x)]
            out[group_id] = self.classify_group_keywords(keywords, contexts)
        return out


def classify_keyword(keyword: str, contexts: Optional[List[str]] = None) -> Dict[str, str]:
    classifier = KeywordClassifier()
    return classifier.classify_group_keywords([keyword], contexts or []).get(
        keyword,
        {"type": "confused", "sense": "分类失败"},
    )
