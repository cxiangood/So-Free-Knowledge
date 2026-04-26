from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from utils import getenv


@dataclass(slots=True)
class LLMConfig:
    api_key: str
    model_id: str
    base_url: str
    temperature: float = 0.2
    max_tokens: int = 512

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str = "",
        model_id: str = "",
        base_url: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> "LLMConfig":
        return cls(
            api_key=(api_key or getenv("LLM_API_KEY", "")).strip(),
            model_id=(model_id or getenv("LLM_MODEL_ID", "")).strip(),
            base_url=(base_url or getenv("LLM_BASE_URL", "")).strip(),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    def missing_fields(self) -> list[str]:
        missing = []
        if not self.api_key:
            missing.append("llm_api_key")
        if not self.model_id:
            missing.append("llm_model_id")
        if not self.base_url:
            missing.append("llm_base_url")
        return missing


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def build_reply(self, system_prompt: str, user_message: str) -> str:
        missing_fields = self.config.missing_fields()
        if missing_fields:
            return "LLM 配置不完整，缺少: " + ", ".join(missing_fields)

        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            return f"LLM 调用失败: {exc}"
        except ValueError:
            return "LLM 返回内容解析失败。"

        content = extract_llm_text(data)
        if content:
            return content
        return "LLM 返回为空。"


def extract_llm_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        return "".join(text_parts).strip()

    return ""
