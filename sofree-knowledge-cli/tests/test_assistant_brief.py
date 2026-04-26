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


def test_build_personal_brief_outputs_doc_and_card():
    report = build_personal_brief(documents=[{"doc_id": "d1", "title": "测试文档", "summary": "摘要"}])
    assert report["doc_markdown"].startswith("# 个人助理聚合简报")
    assert report["card"]["header"]["title"]["content"] == "个人助理聚合建议"
