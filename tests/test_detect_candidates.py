from __future__ import annotations

from local_pipeline.core.detect import detect_candidates


def test_detect_candidates_only_scores_last_message_with_llm(monkeypatch) -> None:
    def _fake_reply(self, system_prompt: str, user_message: str) -> str:
        del self, system_prompt, user_message
        return (
            '{"novelty": 0.9, "actionability": 0.8, "impact": 0.7, '
            '"emotion": 0.4, "reasons": ["contextual-action"], "evidence": "需要安排修复"}'
        )

    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: [])
    monkeypatch.setattr("llm.client.LLMClient.build_reply", _fake_reply)

    result = detect_candidates([
        "[Alice] 大家先看这个问题",
        "[Bob] 我觉得这个点需要安排修复",
    ], candidate_threshold=0.45)

    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.content == "我觉得这个点需要安排修复"
    assert cand.score_breakdown["actionability"] == 0.8
    assert cand.reasons == ["contextual-action"]


def test_detect_candidates_falls_back_when_llm_returns_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: [])
    monkeypatch.setattr("llm.client.LLMClient.build_reply", lambda self, s, u: "not-json")

    result = detect_candidates([
        "[Alice] 这个需求要不要今天完成？",
    ], candidate_threshold=0.2)

    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.score_breakdown["actionability"] > 0.0


def test_detect_candidates_falls_back_when_llm_config_missing(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: ["llm_api_key"])

    result = detect_candidates([
        "[Alice] 建议大家今天完成这个改进",
    ], candidate_threshold=0.2)

    assert len(result.candidates) == 1
    assert result.candidates[0].score_total >= 0.2


def test_detect_candidates_returns_empty_when_current_is_noise() -> None:
    result = detect_candidates([
        "[Alice] 大家先看下",
        "[Bob] this message was recalled",
    ])

    assert result.candidates == []
