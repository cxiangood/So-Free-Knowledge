from sofree_knowledge.assistant_brief import build_personal_brief


def test_build_personal_brief_ranks_docs_with_urgency_and_recommend():
    documents = [
        {"doc_id": "d1", "title": "鍙戝竷娴佺▼", "summary": "涓婄嚎鍓嶅鎵逛笌鍥炴粴"},
        {"doc_id": "d2", "title": "鍛ㄤ細绾", "summary": "渚嬭鍚屾"},
    ]
    access = [
        {"doc_id": "d1", "user_id": "u1", "action": "view", "count": 5},
        {"doc_id": "d1", "user_id": "u1", "action": "edit", "count": 1},
    ]
    messages = [
        {"message_id": "m1", "chat_id": "oc_x", "content": "鍙戝竷浠婂ぉ鎴锛屽敖蹇‘璁ゅ洖婊氶妗?},
        {"message_id": "m2", "chat_id": "oc_x", "content": "鍛ㄤ細鏃堕棿鏀瑰埌鏄庡ぉ"},
    ]
    knowledge = [
        {"id": "k1", "title": "鍙戝竷瑙勮寖", "content": "鍙戝竷闇€瑕佸鎵广€佺伆搴﹀拰鍥炴粴绛栫暐"},
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
        documents=[{"doc_id": "d1", "title": "娴嬭瘯鏂囨。", "summary": "寰呭姙锛氫粖鏅氬畬鎴?}],
        messages=[{"message_id": "m1", "chat_id": "oc_x", "content": "瀹㈡埛闇€姹備粖鏅氭埅姝?}],
        user_profile={
            "persona": "浜у搧缁忕悊",
            "role": "PM",
            "businesses": ["A澧為暱", "B浜や粯"],
            "interests": ["瀹㈡埛", "闇€姹?],
        },
        schedule={
            "mode": "hybrid",
            "weekly_brief_cron": "0 10 * * MON",
            "nightly_interest_cron": "0 21 * * *",
        },
    )

    assert report["doc_markdown"].startswith("# 涓汉鍔╃悊鑱氬悎绠€鎶?)
    assert report["card"]["header"]["title"]["content"] == "涓汉鍔╃悊鑱氬悎寤鸿"
    assert report["interest_card"]["header"]["title"]["content"] == "缇よ亰鍏磋叮娑堟伅姹囨€?
    assert report["documents"][0]["urgency_stars"] >= 1
    assert report["schedule"]["mode"] == "hybrid"
    assert report["runtime_plan"]["cron_jobs"][0]["job_name"] == "assistant_weekly_brief"


def test_interest_digest_filters_noise_messages():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "鍙戝竷璁″垝", "summary": "浠婃櫄鍙戝竷"}],
        messages=[
            {"message_id": "m1", "chat_id": "oc_x", "content": "LLM 璋冪敤澶辫触: 429 Too Many Requests"},
            {"message_id": "m2", "chat_id": "oc_x", "content": "[PLAN_COMMAND] command: create_draft"},
            {"message_id": "m3", "chat_id": "oc_x", "content": "瀹㈡埛闇€姹傚彉鏇达紝浠婃櫄涓婄嚎椋庨櫓杈冮珮锛岄渶瑕佺‘璁ゅ洖婊氭柟妗?},
        ],
        user_profile={"interests": ["闇€姹?, "涓婄嚎", "椋庨櫓", "瀹㈡埛"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert items[0]["message_id"] == "m3"


def test_interest_digest_uses_rewritten_summary_instead_of_raw_prompt():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "鍙戝竷璁″垝", "summary": "浠婃櫄鍙戝竷"}],
        messages=[
            {
                "message_id": "m9",
                "chat_id": "oc_x",
                "content": "@SoFree 璇风敓鎴愪竴浠藉崍楗鍒掞紝瑕佹眰濡備笅锛?. 鍖呭惈3-5涓€夐」 2. 缁欏仛娉?3. 缁欒惀鍏讳寒鐐?,
                "openclaw_summary": "鍗堥キ璁″垝闇€姹傦細闇€瑕?-5涓仴搴疯彍鍝佸苟闄勫仛娉曚笌钀ュ吇浜偣",
            }
        ],
        user_profile={"interests": ["闇€姹?, "瀹㈡埛"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert "瑕佹眰濡備笅" not in items[0]["summary"]
    assert "鍗堥キ璁″垝闇€姹? in items[0]["summary"]


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


def test_interest_card_contains_message_hyperlink_without_hit_or_ids():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Plan summary"}],
        messages=[{"message_id": "om_1", "chat_id": "oc_test", "content": "瀹㈡埛闇€姹備粖鏅氭埅姝紝鏈変笂绾块闄?}],
        user_profile={"interests": ["闇€姹?, "涓婄嚎", "椋庨櫓", "瀹㈡埛"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "瀹㈡埛闇€姹備粖鏅氭埅姝紝鏈変笂绾块闄? in content
    assert "鍛戒腑:" not in content
    assert "msg:" not in content
    assert "chat:" not in content


def test_interest_digest_blocks_openclaw_garbage_message():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_x1",
                "chat_id": "oc_test",
                "content": "鑻ョ户缁彂甯冩绫昏繚瑙勫唴瀹癸紝灏嗘棤娉曚负浣犳彁渚涘悗缁湇鍔★紙鍛戒腑: 鍙戝竷锛?,
                "openclaw_is_garbage": True,
            }
        ],
        user_profile={"interests": ["鍙戝竷", "闇€姹?]},
    )
    assert report["interest_digest"]["items"] == []


def test_interest_digest_blocks_negative_context_without_openclaw_flags():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_x100b518c",
                "chat_id": "oc_f482e00e55461a4d343f21334c9a96d7",
                "content": "鑻ョ户缁彂甯冩绫昏繚瑙勫唴瀹癸紝灏嗘棤娉曚负浣犳彁渚涘悗缁湇鍔★紙鍛戒腑: 鍙戝竷锛?[chat:oc_f482e00e55461a4d343f21334c9a96d7 | msg:om_x100b518c]",
            }
        ],
        user_profile={"interests": ["鍙戝竷", "涓婄嚎"]},
    )
    assert report["interest_digest"]["items"] == []


def test_interest_card_uses_chat_open_applink_with_message_id():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_123",
                "chat_id": "oc_chat",
                "content": "瀹㈡埛闇€姹備粖澶╂埅姝紝鍙戝竷椋庨櫓杈冮珮",`r`n                "message_url": "https://applink.feishu.cn/client/chat/open?chatId=oc_chat&openChatId=oc_chat&openMessageId=om_123",
            }
        ],
        user_profile={"interests": ["瀹㈡埛", "闇€姹?, "鍙戝竷"]},
    )
    items = report["interest_digest"]["items"]
    assert len(items) == 1
    assert items[0]["message_url"].startswith("https://applink.feishu.cn/client/chat/open?")
    assert "openChatId=oc_chat" in items[0]["message_url"]
    assert "openMessageId=om_123" in items[0]["message_url"]
    content = report["interest_card"]["elements"][0]["content"]
    assert "openMessageId=om_123" in content


def test_interest_card_displays_from_sender_name():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_9",
                "chat_id": "oc_chat",
                "content": "Need review before release today",
                "sender_name": "Alice",
            }
        ],
        user_profile={"interests": ["review", "release"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "From锛欰lice" in content


def test_interest_card_keeps_user_mentions_in_summary():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Release Plan", "summary": "Plan summary"}],
        messages=[
            {
                "message_id": "om_m1",
                "chat_id": "oc_chat",
                "content": "@Alice 璇蜂粖澶╁唴纭鍙戝竷鍥炴粴鏂规",
            }
        ],
        user_profile={"interests": ["鍙戝竷", "鍥炴粴"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "@Alice" in content


def test_interest_card_strips_hit_and_ref_artifacts_from_summary():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Any", "summary": "Any"}],
        messages=[
            {
                "message_id": "om_art",
                "chat_id": "oc_chat",
                "content": "锛堝懡涓? 涓婄嚎锛?[chat:oc_chat | msg:om_art] 鍘熸秷鎭?,
            }
        ],
        user_profile={"interests": ["涓婄嚎"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "鍛戒腑:" not in content
    assert "chat:" not in content
    assert "msg:" not in content


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


def test_interest_card_from_falls_back_to_open_id_when_name_missing():
    report = build_personal_brief(
        documents=[{"doc_id": "d1", "title": "Any", "summary": "Any"}],
        messages=[
            {
                "message_id": "om_sender",
                "chat_id": "oc_chat",
                "content": "涓婄嚎鎻愰啋",
                "sender": {"sender_id": {"open_id": "ou_abc"}},
            }
        ],
        user_profile={"interests": ["涓婄嚎"]},
    )
    content = report["interest_card"]["elements"][0]["content"]
    assert "From锛歰u_abc" in content

