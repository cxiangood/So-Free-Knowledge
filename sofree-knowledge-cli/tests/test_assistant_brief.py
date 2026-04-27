from sofree_knowledge.assistant_brief import build_personal_brief


def test_build_personal_brief_ranks_docs_with_urgency_and_recommend():
    documents = [
        {"doc_id": "d1", "title": "发布流程", "summary": "上线前审批与回滚"},
        {"doc_id": "d2", "title": "周会纪要", "summary": "例行同步"},
    ]
    access = [
        {"doc_id": "d1", "user_id": "u1", "action": "view", "count": 5},
        {"doc_id": "d1", "user_id": "u1", "action": "edit", "count": 1},
    ]
    messages = [
        {"message_id": "m1", "chat_id": "oc_x", "content": "发布今天截止，尽快确认回滚预案"},
        {"message_id": "m2", "chat_id": "oc_x", "content": "周会时间改到明天"},
    ]
    knowledge = [
        {"id": "k1", "title": "发布规范", "content": "发布需要审批、灰度和回滚策略"},
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
    assert report["documents"][0]["related_messages"]
    assert report["documents"][0]["related_knowledge"]


def test_build_personal_brief_outputs_doc_card_interest_and_schedule():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "测试文档", "summary": "待办：今晚完成"}],
        messages=[{"message_id": "m1", "chat_id": "oc_x", "content": "客户需求今晚截止"}],
        user_profile={
            "persona": "产品经理",
            "role": "PM",
            "businesses": ["A增长", "B交付"],
            "interests": ["客户", "需求"],
        },
        schedule={
            "mode": "hybrid",
            "weekly_brief_cron": "0 10 * * MON",
            "nightly_interest_cron": "0 21 * * *",
        },
    )

    assert report["doc_markdown"].startswith("# 个人助理聚合简报")
    assert report["card"]["header"]["title"]["content"] == "个人助理聚合建议"
    assert report["interest_card"]["header"]["title"]["content"] == "群聊兴趣消息汇总"
    assert report["documents"][0]["urgency_stars"] >= 1
    assert report["schedule"]["mode"] == "hybrid"
    assert report["runtime_plan"]["cron_jobs"][0]["job_name"] == "assistant_weekly_brief"


def test_interest_digest_filters_noise_messages():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "发布计划", "summary": "今晚发布"}],
        messages=[
            {"message_id": "m1", "chat_id": "oc_x", "content": "LLM 调用失败: 429 Too Many Requests"},
            {"message_id": "m2", "chat_id": "oc_x", "content": "[PLAN_COMMAND] command: create_draft"},
            {"message_id": "m3", "chat_id": "oc_x", "content": "客户需求变更，今晚上线风险较高，需要确认回滚方案"},
        ],
        user_profile={"interests": ["需求", "上线", "风险", "客户"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert items[0]["message_id"] == "m3"


def test_interest_digest_uses_rewritten_summary_instead_of_raw_prompt():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "发布计划", "summary": "今晚发布"}],
        messages=[
            {
                "message_id": "m9",
                "chat_id": "oc_x",
                "content": "@SoFree 请生成一份午饭计划，要求如下：1. 包含3-5个选项 2. 给做法 3. 给营养亮点",
                "openclaw_summary": "午饭计划需求：需要3-5个健康菜品并附做法与营养亮点",
            }
        ],
        user_profile={"interests": ["需求", "客户"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert "要求如下" not in items[0]["summary"]
    assert "午饭计划需求" in items[0]["summary"]


def test_summary_card_contains_hyperlink_when_url_available():
    report = build_personal_brief(
        documents=[
            {
                "doc_id": "d1",
                "title": "Release Plan",
                "summary": "Plan summary",
                "url": "https://foo.feishu.cn/docx/abc123",
            }
        ]
    )
    content = report["card"]["elements"][0]["content"]
    assert "[Release Plan](https://foo.feishu.cn/docx/abc123)" in content


def test_interest_card_contains_message_hyperlink():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Plan summary"}],
        messages=[{"message_id": "om_1", "chat_id": "oc_test", "content": "客户需求今晚截止，有上线风险"}],
        user_profile={"interests": ["需求", "上线", "风险", "客户"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "客户需求今晚截止，有上线风险" in content
    assert "chat:oc_test" in content
    assert "msg:om_1" in content


def test_interest_digest_blocks_moderation_text_even_with_hit_term():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_x1",
                "chat_id": "oc_test",
                "content": "若继续发布此类违规内容，将无法为你提供后续服务（命中: 发布）",
            }
        ],
        user_profile={"interests": ["发布", "需求"]},
    )
    assert report["interest_digest"]["items"] == []


def test_interest_card_uses_chat_open_applink_with_message_id():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_123",
                "chat_id": "oc_chat",
                "content": "客户需求今天截止，发布风险较高",
            }
        ],
        user_profile={"interests": ["客户", "需求", "发布"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert items[0]["message_url"].startswith("https://applink.feishu.cn/client/chat/open?")
    assert "openChatId=oc_chat" in items[0]["message_url"]
    assert "openMessageId=om_123" in items[0]["message_url"]
    content = report["interest_card"]["elements"][0]["content"]
    assert "openMessageId=om_123" in content
