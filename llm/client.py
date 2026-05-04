from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, TypeVar

import requests
from pydantic import BaseModel, Field

from utils import getenv


LOGGER = logging.getLogger(__name__)
RATE_LIMIT_LOCK = RLock()
RATE_LIMIT_ACTIVE = False
RATE_LIMIT_UNTIL = 0.0
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
RATE_LIMIT_SAFETY_SECONDS = 0.0


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
    participants: list[str] = Field(default_factory=list)
    times: str
    locations: str
    topic_focus: str = ""
    message_role: str = ""
    context_relation: str = ""
    context_evidence: list[str] = Field(default_factory=list)
    decision_signals: dict[str, float] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)


class RouteItem(BaseModel):
    target_pool: Literal["knowledge", "task", "observe"]
    reason_codes: list[str] = Field(default_factory=list)


class RouteOutput(BaseModel):
    routes: list[RouteItem] = Field(default_factory=list, min_length=1, max_length=3)


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


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, requests.Timeout):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "timed out" in str(exc).lower()


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "ratelimit" in text or "too many requests" in text or "rpm" in text


def _is_endpoint_restricted_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {401, 403}:
        return True
    text = str(exc).lower()
    return "permission" in text or "forbidden" in text or "unauthorized" in text


def _log_structured_error(*, exc: Exception, model_id: str, schema_name: str) -> None:
    if _is_timeout_error(exc):
        LOGGER.warning(
            "structured llm timeout; fallback expected (model=%s, schema=%s): %s",
            model_id,
            schema_name,
            exc,
        )
        return
    if _is_rate_limit_error(exc):
        LOGGER.warning(
            "structured llm rate limited; fallback expected (model=%s, schema=%s): %s",
            model_id,
            schema_name,
            exc,
        )
        return
    if _is_endpoint_restricted_error(exc):
        LOGGER.warning(
            "structured llm endpoint restricted; fallback expected (model=%s, schema=%s): %s",
            model_id,
            schema_name,
            exc,
        )
        return
    LOGGER.error(
        "structured llm invocation failed with unknown cause (model=%s, schema=%s)",
        model_id,
        schema_name,
        exc_info=True,
    )


def _cooldown_seconds() -> float:
    return max(0.0, float(RATE_LIMIT_COOLDOWN_SECONDS) + float(RATE_LIMIT_SAFETY_SECONDS))


def _enter_rate_limit_cooldown(seconds: float | None = None) -> None:
    global RATE_LIMIT_ACTIVE, RATE_LIMIT_UNTIL
    wait_seconds = _cooldown_seconds() if seconds is None else max(0.0, float(seconds))
    now = time.monotonic()
    with RATE_LIMIT_LOCK:
        if RATE_LIMIT_ACTIVE and RATE_LIMIT_UNTIL > now:
            return
        RATE_LIMIT_ACTIVE = True
        RATE_LIMIT_UNTIL = now + wait_seconds
    LOGGER.warning("llm global rate-limit cooldown entered for %.2fs", wait_seconds)


def _wait_if_rate_limited() -> None:
    global RATE_LIMIT_ACTIVE, RATE_LIMIT_UNTIL
    while True:
        with RATE_LIMIT_LOCK:
            if not RATE_LIMIT_ACTIVE:
                return
            remaining = RATE_LIMIT_UNTIL - time.monotonic()
            if remaining <= 0:
                RATE_LIMIT_ACTIVE = False
                RATE_LIMIT_UNTIL = 0.0
                return
        LOGGER.info("llm global rate-limit cooldown active, blocking for %.2fs", remaining)
        time.sleep(remaining)


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
    schema_name = getattr(schema, "__name__", "unknown")
    while True:
        _wait_if_rate_limited()
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
            if _is_rate_limit_error(exc):
                _enter_rate_limit_cooldown()
                continue
            _log_structured_error(exc=exc, model_id=config.model_id, schema_name=schema_name)
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
        if self.config.extra_body:
            payload.update(self.config.extra_body)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        while True:
            _wait_if_rate_limited()
            try:
                response = requests.post(endpoint, json=payload, headers=headers, timeout=600)
                response.raise_for_status()
                data = response.json()
            except requests.Timeout as exc:
                LOGGER.warning("chat llm timeout (model=%s): %s", self.config.model_id, exc)
                return f"LLM 调用失败: {exc}"
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 429:
                    LOGGER.warning("chat llm rate limited (model=%s, status=429): %s", self.config.model_id, exc)
                    _enter_rate_limit_cooldown()
                    continue
                if status in {401, 403}:
                    LOGGER.warning("chat llm endpoint restricted (model=%s, status=%s): %s", self.config.model_id, status, exc)
                else:
                    LOGGER.error("chat llm http error (model=%s, status=%s)", self.config.model_id, status, exc_info=True)
                return f"LLM 调用失败: {exc}"
            except requests.RequestException as exc:
                LOGGER.error("chat llm request failed with unknown cause (model=%s)", self.config.model_id, exc_info=True)
                return f"LLM 调用失败: {exc}"
            except ValueError:
                LOGGER.error("chat llm response json parse failed (model=%s)", self.config.model_id, exc_info=True)
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
