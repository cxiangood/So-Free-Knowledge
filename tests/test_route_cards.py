from __future__ import annotations

from local_pipeline.core.route import route_cards
from local_pipeline.shared.models import LiftedCard


def _card(**overrides) -> LiftedCard:
    base = {
        "card_id": "card-1",
        "candidate_id": "cand-1",
        "title": "修复导出失败并沉淀规则",
        "summary": "导出失败需要修复，同时权限规则可沉淀。",
        "problem": "导出失败影响用户使用。",
        "suggestion": "今天完成修复并更新文档。",
        "target_audience": "团队成员",
        "evidence": ["导出失败和权限改动有关"],
        "tags": ["actionable-signal", "novel-content"],
        "confidence": 0.86,
        "suggested_target": "task",
        "source_message_ids": ["current"],
        "message_role": "update",
        "decision_signals": {
            "novelty_score": 0.8,
            "actionability_score": 0.88,
            "impact_score": 0.74,
            "has_question": 0.0,
            "has_action_hint": 1.0,
        },
        "missing_fields": [],
    }
    base.update(overrides)
    return LiftedCard(**base)


def test_route_cards_rule_can_expand_one_card_to_multiple_targets(monkeypatch) -> None:
    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: ["llm_api_key"])

    decisions = route_cards([_card()])

    assert [item.target_pool for item in decisions] == ["knowledge", "task"]
    assert decisions[0].card_id == "card-1"
    assert "knowledge-semantic-signal" in decisions[0].reason_codes
    assert "suggested-task" in decisions[1].reason_codes


def test_route_cards_llm_routes_are_deduplicated_and_ordered(monkeypatch) -> None:
    from llm.client import RouteItem, RouteOutput

    monkeypatch.setattr("llm.client.LLMConfig.missing_fields", lambda self: [])
    monkeypatch.setattr(
        "llm.client.invoke_structured",
        lambda **kwargs: RouteOutput(
            routes=[
                RouteItem(target_pool="observe", reason_codes=["weak-or-unclear-signal"]),
                RouteItem(target_pool="task", reason_codes=["action-signal"]),
                RouteItem(target_pool="task", reason_codes=["task-semantic-signal"]),
            ]
        ),
    )

    decisions = route_cards([_card(card_id="card-2")])

    assert [item.target_pool for item in decisions] == ["task", "observe"]
    assert decisions[0].reason_codes == ["action-signal", "task-semantic-signal"]
