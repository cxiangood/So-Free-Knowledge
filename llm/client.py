from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import requests
from pydantic import BaseModel, Field

from utils import getenv


class DetectScores(BaseModel):
    novelty: float = Field(ge=0.0, le=100.0)
    actionability: float = Field(ge=0.0, le=100.0)
    impact: float = Field(ge=0.0, le=100.0)
    emotion: float = Field(ge=0.0, le=100.0)


class LiftParts(BaseModel):
    title: str
    summary: str
    suggestion: str
    problem: str
    names: list[str] = Field(default_factory=list)


class RouteOutput(BaseModel):
    target_pool: Literal["knowledge", "task", "observe"]
    reason_codes: list[str] = Field(default_factory=list)


class ObserveQuestionOutput(BaseModel):
    is_question: bool | None = None
    need_reply: bool | None = None
    confidence: float | None = None


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


@dataclass(slots=True)
class LLMConfig:
    api_key: str
    model_id: str
    base_url: str
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 1.0

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str = "",
        model_id: str = "",
        base_url: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ) -> "LLMConfig":
        return cls(
            api_key=(api_key or getenv("LLM_API_KEY", "")).strip(),
            model_id=(model_id or getenv("LLM_MODEL_ID", "")).strip(),
            base_url=(base_url or getenv("LLM_BASE_URL", "")).strip(),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
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


def invoke_structured(
    *,
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    schema: type[StructuredModel],
    method: str = "json_mode",
) -> StructuredModel | None:
    if config.missing_fields():
        return None
    try:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=config.model_id,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        structured_model = model.with_structured_output(schema, method=method)
        response = structured_model.invoke(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
    except Exception:
        return None

    if isinstance(response, schema):
        return response
    if isinstance(response, dict):
        try:
            return schema(**response)
        except Exception:
            return None
    return None


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
            "top_p": self.config.top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=600)
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
