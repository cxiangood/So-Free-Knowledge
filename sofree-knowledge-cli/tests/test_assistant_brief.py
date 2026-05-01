from sofree_knowledge.assistant_brief import build_personal_brief


def test_build_personal_brief_ranks_docs_with_urgency_and_recommend():
    documents = [
        {"doc_id": "d1", "title": "Release Flow", "summary": "Pre-release checklist"},
        {"doc_id": "d2", "title": "Weekly Notes", "summary": "Routine sync"},
    ]
    access = [
        {"doc_id": "d1", "user_id": "u1", "action": "view", "count": 5},
        {"doc_id": "d1", "user_id": "u1", "action": "edit", "count": 1},
    ]
    messages = [
        {"message_id": "m1", "chat_id": "oc_x", "content": "Release is due today, confirm rollback plan"},
        {"message_id": "m2", "chat_id": "oc_x", "content": "Weekly sync moved to tomorrow"},
    ]
    knowledge = [
        {"id": "k1", "title": "Release Guide", "content": "Needs review and rollback strategy"},
    ]

    report = build_personal_brief(
        documents=documents,
        access_records=access,
        messages=messages,
        knowledge_items=knowledge,
        target_user_id="u1",
    )

    assert report["summary"]["doc_count"] == 2
    assert report["documents"][0]["doc_id"] == "d1"
    assert report["documents"][0]["urgency_score"] >= report["documents"][1]["urgency_score"]
    assert report["documents"][0]["recommend_score"] >= report["documents"][1]["recommend_score"]
    assert report["documents"][0]["retrieval"]["dual_tower_ready"] is True
    assert report["retrieval_plan"]["dual_tower_ready"] is True


def test_build_personal_brief_emits_dual_tower_text_payload():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Flow", "summary": "Rollback checklist for release"}],
        messages=[{"message_id": "m1", "chat_id": "oc_x", "content": "Release due today"}],
        user_profile={
            "role": "PM",
            "persona": "冷静分析",
            "businesses": ["Release"],
            "interests": ["risk", "release"],
        },
        dual_tower_config={"enabled": True, "embedding_model": "text-embedding-3-large", "top_k": 20},
    )

    retrieval = report["documents"][0]["retrieval"]
    assert "role: PM" in retrieval["user_tower_text"]
    assert "title: Release Flow" in retrieval["content_tower_text"]
    assert retrieval["dual_tower_score"] > 0
    assert report["retrieval_plan"]["dual_tower_enabled"] is True
    assert report["retrieval_plan"]["embedding_model"] == "text-embedding-3-large"


def test_build_personal_brief_can_fallback_to_openclaw_strategy():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Flow", "summary": "Rollback checklist for release"}],
        messages=[
            {
                "message_id": "m1",
                "chat_id": "oc_x",
                "content": "Release due today",
                "openclaw_interest_relevant": True,
            }
        ],
        user_profile={"role": "PM", "interests": ["release"]},
        dual_tower_config={"enabled": False},
    )

    retrieval = report["documents"][0]["retrieval"]
    assert retrieval["strategy"] == "openclaw_fallback"
    assert retrieval["dual_tower_ready"] is False
    assert retrieval["user_tower_text"] == ""
    assert report["retrieval_plan"]["strategy"] == "openclaw_fallback"
    assert report["retrieval_plan"]["dual_tower_enabled"] is False


def test_build_personal_brief_applies_trained_dual_tower_model(tmp_path):
    model_file = tmp_path / "dual_tower_model.json"
    model_file.write_text(
        json.dumps(
            {
                "model_type": "dual_tower_baseline_term_weight",
                "token_weights": {"release": 2.0, "rollback": 1.5},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Flow", "summary": "Rollback checklist for release"}],
        messages=[{"message_id": "m1", "chat_id": "oc_x", "content": "Release due today"}],
        user_profile={"role": "PM", "persona": "冷静分析", "businesses": ["Release"], "interests": ["release", "risk"]},
        dual_tower_config={
            "enabled": True,
            "embedding_model": "text-embedding-3-large",
            "model_file": str(model_file),
        },
    )

    retrieval = report["documents"][0]["retrieval"]
    assert retrieval["strategy"] == "dual_tower_trained"
    assert retrieval["model_applied"] is True
    assert retrieval["trained_dual_tower_score"] > retrieval["dual_tower_score"]
    assert report["retrieval_plan"]["strategy"] == "dual_tower_trained"
    assert report["retrieval_plan"]["model_file"] == str(model_file)


def test_interest_digest_filters_noise_messages():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Tonight release"}],
        messages=[
            {"message_id": "m1", "chat_id": "oc_x", "content": "LLM 调用失败: 429 Too Many Requests"},
            {"message_id": "m2", "chat_id": "oc_x", "content": "[PLAN_COMMAND] command: create_draft"},
            {"message_id": "m3", "chat_id": "oc_x", "content": "Customer asks for change, release risk is high tonight"},
        ],
        user_profile={"interests": ["customer", "release", "risk"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert items[0]["message_id"] == "m3"


def test_interest_digest_applies_interest_filter_judgement_from_message():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "m_filter",
                "chat_id": "oc_x",
                "content": "系统处罚提醒",
                "interest_filter_judgement": {
                    "message_id": "m_filter",
                    "include_in_digest": False,
                    "is_garbage": True,
                    "importance": 0.0,
                    "summary": "",
                },
            }
        ],
        user_profile={"interests": ["发布"]},
    )
    assert report["interest_digest"]["items"] == []


def test_interest_digest_blocks_openclaw_garbage_message():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_x1",
                "chat_id": "oc_test",
                "content": "若继续发布此类违规内容，将无法为你提供后续服务（命中: 发布）",
                "openclaw_is_garbage": True,
            }
        ],
        user_profile={"interests": ["发布", "需求"]},
    )
    assert report["interest_digest"]["items"] == []


def test_interest_digest_blocks_negative_context_without_openclaw_flags():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_x100b518c",
                "chat_id": "oc_f482e00e55461a4d343f21334c9a96d7",
                "content": "若继续发布此类违规内容，将无法为你提供后续服务（命中: 发布） [chat:oc_f482e00e55461a4d343f21334c9a96d7 | msg:om_x100b518c]",
            }
        ],
        user_profile={"interests": ["发布", "上线"]},
    )
    assert report["interest_digest"]["items"] == []


def test_interest_card_uses_message_url_from_upstream():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_123",
                "chat_id": "oc_chat",
                "content": "Customer demand due today, release risk high",
                "message_url": "https://applink.feishu.cn/client/chat/open?chatId=oc_chat&openChatId=oc_chat&openMessageId=om_123",
            }
        ],
        user_profile={"interests": ["customer", "release", "risk"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert items[0]["message_url"] == "https://applink.feishu.cn/client/chat/open?openChatId=oc_chat"
    content = report["interest_card"]["elements"][0]["content"]
    assert "https://applink.feishu.cn/client/chat/open?openChatId=oc_chat" in content


def test_interest_card_contains_no_hit_or_ref_artifacts():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Any", "summary": "Any"}],
        messages=[
            {
                "message_id": "om_art",
                "chat_id": "oc_chat",
                "content": "Feature goes online next week, please re-check (命中: 上线) [chat:oc_chat | msg:om_art]",
            }
        ],
        user_profile={"interests": ["上线"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "命中:" not in content
    assert "chat:" not in content
    assert "msg:" not in content


def test_interest_digest_prefers_raw_text_when_summary_has_structured_artifacts():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Any", "summary": "Any"}],
        messages=[
            {
                "message_id": "om_pref",
                "chat_id": "oc_chat",
                "content": "我们的关键词提取功能预计下周上线哈，你赶快把相关的内容再检查一遍",
                "openclaw_summary": "我们的关键词提取功能预计下周上线哈，你赶快把相关的内容再检查一遍（命中: 上线） [chat:oc_chat | msg:om_pref]",
            }
        ],
        user_profile={"interests": ["上线"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "命中:" not in content
    assert "chat:" not in content
    assert "我们的关键词提取功能预计下周上线哈，你赶快把相关的内容再检查一遍" in content


def test_interest_card_displays_from_sender_name():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_9",
                "chat_id": "oc_chat",
                "content": "Need review before release today",
                "sender_name": "Xiao Li",
            }
        ],
        user_profile={"interests": ["review", "release"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "From：Xiao Li" in content


def test_interest_card_from_falls_back_to_open_id_when_name_missing():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Any", "summary": "Any"}],
        messages=[
            {
                "message_id": "om_sender",
                "chat_id": "oc_chat",
                "content": "上线提醒",
                "sender": {"sender_id": {"open_id": "ou_abc"}},
            }
        ],
        user_profile={"interests": ["上线"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "From：ou_abc" in content


def test_interest_card_keeps_user_mentions_in_summary():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_m1",
                "chat_id": "oc_chat",
                "content": "@Alice please confirm rollback plan today",
            }
        ],
        user_profile={"interests": ["发布", "回滚"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "@Alice" in content


def test_interest_digest_accepts_at_user_without_interest_keyword_hit():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Any", "summary": "Any"}],
        messages=[
            {
                "message_id": "om_at_user",
                "chat_id": "oc_chat",
                "content": "@Bob please confirm this today",
            }
        ],
        user_profile={"interests": ["release", "risk"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert items[0]["message_id"] == "om_at_user"


def test_interest_digest_filters_low_information_placeholder_mentions():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Any", "summary": "Any"}],
        messages=[
            {"message_id": "m1", "chat_id": "oc_chat", "content": "@1 回答一下", "sender_name": "聂铭浚"},
            {"message_id": "m2", "chat_id": "oc_chat", "content": "@1 你要@2 ，@3 你要@4", "sender_name": "聂铭浚"},
            {"message_id": "m3", "chat_id": "oc_chat", "content": "和@1 讨论一下项目形态", "sender_name": "聂铭浚"},
            {
                "message_id": "m4",
                "chat_id": "oc_chat",
                "content": "@1 你好呀～有什么需要我帮忙的吗",
                "sender_name": "cli_a960422c68f81cc8",
            },
        ],
        user_profile={"interests": ["release", "risk"]},
    )
    assert report["interest_digest"]["items"] == []


def test_interest_digest_accepts_at_all_without_interest_keyword_hit():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Any", "summary": "Any"}],
        messages=[
            {
                "message_id": "om_at_all",
                "chat_id": "oc_chat",
                "content": "@all please check update",
            }
        ],
        user_profile={"interests": ["release", "risk"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert items[0]["message_id"] == "om_at_all"
import json
