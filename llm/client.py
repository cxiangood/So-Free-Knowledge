from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import requests
from pydantic import BaseModel, Field

from utils import getenv


LOGGER = logging.getLogger(__name__)


def _log_llm_metric(
    *,
    kind: str,
    model_id: str,
    elapsed_ms: float,
    success: bool,
    schema_name: str = "",
    usage: dict[str, Any] | None = None,
    error: str = "",
) -> None:
    payload = {
        "kind": kind,
        "model": model_id,
        "schema": schema_name,
        "elapsed_ms": round(elapsed_ms, 2),
        "success": success,
        "usage": usage or {},
        "error": error,
    }
    LOGGER.info("llm_metric %s", payload)


class DetectScores(BaseModel):
    novelty: float = Field(ge=0.0, le=100.0)
    actionability: float = Field(ge=0.0, le=100.0)
    impact: float = Field(ge=0.0, le=100.0)
    emotion: float = Field(ge=0.0, le=100.0)


class DetectMeaningful(BaseModel):
    meaningful: bool


class DetectValueScore(BaseModel):
    value_score: float = Field(ge=0.0, le=100.0)


class LiftParts(BaseModel):
    title: str
    summary: str
    suggestion: str
    problem: str
    names: list[str] = Field(default_factory=list)
    topic_focus: str = ""
    message_role: str = ""
    context_relation: str = ""
    context_evidence: list[str] = Field(default_factory=list)
    decision_signals: dict[str, float] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)


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
    extra_body: dict[str, Any] = field(default_factory=dict)

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
        extra_body: dict[str, Any] | None = None,
    ) -> "LLMConfig":
        return cls(
            api_key=(api_key or getenv("LLM_API_KEY", "")).strip(),
            model_id=(model_id or getenv("LLM_MODEL_ID", "")).strip(),
            base_url=(base_url or getenv("LLM_BASE_URL", "")).strip(),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_body=dict(extra_body or {}),
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
    start = time.perf_counter()
    schema_name = getattr(schema, "__name__", "")
    try:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=config.model_id,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            extra_body=config.extra_body or None,
        )
        structured_model = model.with_structured_output(schema, method=method)
        response = structured_model.invoke(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _log_llm_metric(
            kind="structured",
            model_id=config.model_id,
            schema_name=schema_name,
            elapsed_ms=elapsed_ms,
            success=False,
            error=type(exc).__name__,
        )
        LOGGER.exception("structured llm invocation failed")
        return None

    if isinstance(response, schema):
        elapsed_ms = (time.perf_counter() - start) * 1000
        _log_llm_metric(
            kind="structured",
            model_id=config.model_id,
            schema_name=schema_name,
            elapsed_ms=elapsed_ms,
            success=True,
        )
        return response
    if isinstance(response, dict):
        try:
            parsed = schema(**response)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _log_llm_metric(
                kind="structured",
                model_id=config.model_id,
                schema_name=schema_name,
                elapsed_ms=elapsed_ms,
                success=True,
            )
            return parsed
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            _log_llm_metric(
                kind="structured",
                model_id=config.model_id,
                schema_name=schema_name,
                elapsed_ms=elapsed_ms,
                success=False,
                error=type(exc).__name__,
            )
            return None
    elapsed_ms = (time.perf_counter() - start) * 1000
    _log_llm_metric(
        kind="structured",
        model_id=config.model_id,
        schema_name=schema_name,
        elapsed_ms=elapsed_ms,
        success=False,
        error=f"unexpected_response:{type(response).__name__}",
    )
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
        if self.config.extra_body:
            payload.update(self.config.extra_body)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        start = time.perf_counter()
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=600)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            _log_llm_metric(
                kind="chat",
                model_id=self.config.model_id,
                elapsed_ms=elapsed_ms,
                success=False,
                error=type(exc).__name__,
            )
            return f"LLM 调用失败: {exc}"
        except ValueError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            _log_llm_metric(
                kind="chat",
                model_id=self.config.model_id,
                elapsed_ms=elapsed_ms,
                success=False,
                error=type(exc).__name__,
            )
            return "LLM 返回内容解析失败。"

        elapsed_ms = (time.perf_counter() - start) * 1000
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        _log_llm_metric(
            kind="chat",
            model_id=self.config.model_id,
            elapsed_ms=elapsed_ms,
            success=True,
            usage=usage,
        )
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
