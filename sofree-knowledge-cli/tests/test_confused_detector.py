import json

from sofree_knowledge.confused_detector import (
    build_confused_judge_prompt,
    detect_confused_candidates,
    format_inline_explanation,
    parse_confused_judgement,
)


def test_detect_confused_candidates_with_reply_phrase():
    messages = [
        {"message_id": "m1", "sender": {"id": "u1"}, "content": "这个流程先走审批再发布"},
        {
            "message_id": "m2",
            "sender": {"id": "u2"},
            "parent_id": "m1",
            "content": "什么？没懂",
        },
    ]

    candidates = detect_confused_candidates(messages=messages, target_message_id="m1")

    assert len(candidates) == 1
    assert candidates[0]["target_message_id"] == "m1"
    assert "confused_phrase" in candidates[0]["triggers"]
    assert "reply_or_thread_followup" in candidates[0]["triggers"]


def test_detect_confused_candidates_with_reaction():
    messages = [{"message_id": "m1", "sender": {"id": "u1"}, "content": "A/B 测试先看点击率"}]
    reactions = [{"message_id": "m1", "reaction_key": "question", "user_id": "u2"}]

    candidates = detect_confused_candidates(messages=messages, reactions=reactions)

    assert len(candidates) == 1
    assert candidates[0]["target_message_id"] == "m1"
    assert "confused_reaction" in candidates[0]["triggers"]


def test_detect_confused_candidates_with_question_mark_reaction():
    messages = [{"message_id": "m1", "sender": {"id": "u1"}, "content": "发布前先走审批"}]
    reactions = [{"message_id": "m1", "reaction_key": "？", "user_id": "u2"}]

    candidates = detect_confused_candidates(messages=messages, reactions=reactions)

    assert len(candidates) == 1
    assert "confused_reaction" in candidates[0]["triggers"]


def test_detect_confused_candidates_ignores_emoji_only_reaction():
    messages = [{"message_id": "m1", "sender": {"id": "u1"}, "content": "A/B 测试先看点击率"}]
    reactions = [{"message_id": "m1", "emoji": "❓", "user_id": "u2"}]

    candidates = detect_confused_candidates(messages=messages, reactions=reactions)

    assert candidates == []


def test_detect_confused_candidates_with_single_question_message():
    messages = [
        {"message_id": "m1", "sender": {"id": "u1"}, "content": "先审批再发布"},
        {"message_id": "m2", "sender": {"id": "u2"}, "parent_id": "m1", "content": "？"},
    ]

    candidates = detect_confused_candidates(messages=messages)

    assert len(candidates) == 1
    assert "confused_phrase" in candidates[0]["triggers"]


def test_parse_confused_judgement_and_inline_insert():
    raw = json.dumps(
        {
            "is_confused": True,
            "confidence": 0.91,
            "reason": "追问表达出不理解",
            "micro_explain": "这里是先审批通过，再进入发布步骤",
        },
        ensure_ascii=False,
    )

    judgement = parse_confused_judgement(raw)
    inline_text = format_inline_explanation(judgement["micro_explain"])
    prompt = build_confused_judge_prompt(
        {"target_message_id": "m1", "triggers": ["confused_phrase"], "evidence": [], "context_messages": []}
    )

    assert judgement["is_confused"] is True
    assert inline_text == "（这里是先审批通过，再进入发布步骤）"
    assert "micro_explain" in prompt
