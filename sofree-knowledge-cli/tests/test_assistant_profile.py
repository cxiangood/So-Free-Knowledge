from sofree_knowledge.assistant.profile import suggest_profile_from_online_inputs
from sofree_knowledge.assistant_brief import _normalize_profile


def test_suggest_profile_appends_display_name_to_interest_keywords():
    profile = suggest_profile_from_online_inputs(
        online_inputs={
            "documents": [{"title": "飞书知识推送", "summary": "LLM 产品设计"}],
            "messages": [],
            "knowledge_items": [],
        },
        display_name="曹林江",
    )

    assert "曹林江" in profile["interests"]


def test_normalize_profile_appends_display_name_to_interests():
    normalized = _normalize_profile(
        {
            "display_name": "曹林江",
            "interests": ["飞书", "产品设计"],
        }
    )

    assert normalized["interests"] == ["飞书", "产品设计", "曹林江"]
    assert normalized["display_name"] == "曹林江"
