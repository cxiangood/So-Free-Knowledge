from __future__ import annotations

from llm.client import DetectValueScore
from local_pipeline.core.detect import detect_candidates


def test_detect_candidates_only_checks_last_message_with_llm(monkeypatch) -> None:
    def _fake_structured(**kwargs):
        del kwargs
        return DetectValueScore(value_score=82.0)

    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: [])
    monkeypatch.setattr("llm.client.invoke_structured", _fake_structured)

    result = detect_candidates(
        [
            "[Alice] 大家先看这个问题",
            "[Bob] 我觉得这个点需要安排修复",
        ]
    )

    assert result.value_score == 82.0
    assert result.messages


def test_detect_candidates_falls_back_when_structured_llm_returns_none(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: [])
    monkeypatch.setattr("llm.client.invoke_structured", lambda **kwargs: None)

    result = detect_candidates(
        [
            "[Alice] 这个需求要不要今天完成？",
        ]
    )

    assert result.value_score > 0.0


def test_detect_candidates_falls_back_when_llm_config_missing(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: ["llm_api_key"])

    result = detect_candidates(
        [
            "[Alice] 建议大家今天完成这个改进",
        ]
    )

    assert result.value_score > 0.0


def test_detect_candidates_returns_false_when_current_is_noise() -> None:
    result = detect_candidates(
        [
            "[Alice] 大家先看下",
            "[Bob] this message was recalled",
        ]
    )

    assert result.value_score == 0.0
