import json

from sofree_knowledge.interest_filter import (
    build_interest_filter_prompt,
    parse_interest_filter_judgements,
)


def test_build_interest_filter_prompt_contains_schema_and_payload():
    prompt = build_interest_filter_prompt(
        messages=[{"message_id": "m1", "chat_id": "oc_x", "content": "发布今天截止"}],
        interests=["发布", "风险"],
    )
    assert "include_in_digest" in prompt
    assert "is_garbage" in prompt
    assert "importance" in prompt
    assert "\"message_id\": \"m1\"" in prompt


def test_parse_interest_filter_judgements_normalizes_fields():
    raw = json.dumps(
        [
            {
                "message_id": "m1",
                "include_in_digest": True,
                "is_garbage": False,
                "importance": 0.86,
                "reason": "有明确截止时间",
                "summary": "发布今天截止，需要确认回滚",
            },
            {
                "message_id": "m2",
                "include_in_digest": False,
                "is_garbage": True,
                "importance": 0.05,
                "reason": "系统提醒",
                "summary": "应被清空",
            },
        ],
        ensure_ascii=False,
    )
    parsed = parse_interest_filter_judgements(raw)
    assert len(parsed) == 2
    assert parsed[0]["include_in_digest"] is True
    assert parsed[0]["summary"] == "发布今天截止，需要确认回滚"
    assert parsed[1]["is_garbage"] is True
    assert parsed[1]["summary"] == ""
