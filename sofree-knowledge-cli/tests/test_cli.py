import json

import sofree_knowledge.cli as cli_module
from sofree_knowledge.cli import main


def test_set_knowledge_scope_cli_outputs_json(tmp_path, capsys):
    code = main(["--output-dir", str(tmp_path), "set-knowledge-scope", "oc_test", "chat_only"])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["scope"] == "chat_only"


def test_auth_status_cli_outputs_json(tmp_path, capsys):
    token_file = tmp_path / "missing.json"

    code = main(["auth-status", "--token-file", str(token_file)])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["exists"] is False


def test_auth_login_no_wait_cli_outputs_json(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "start_device_login",
        lambda scope: {
            "flow": "device_code",
            "request": {
                "device_code": "dev123",
                "verification_uri_complete": "https://accounts.feishu.cn/verify",
            },
        },
    )

    code = main(["auth", "login", "--no-wait", "--scope", "im:chat:read"])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["flow"] == "device_code"
    assert out["request"]["device_code"] == "dev123"


def test_auth_login_bootstraps_profile_after_token(monkeypatch, tmp_path, capsys):
    class FakeAuthClient:
        def request(self, method, path, params=None, access_token=None, **kwargs):
            return {"data": {"user": {"name": "曹林江"}}}

        def get_tenant_access_token(self):
            return "tenant_token"

    monkeypatch.setattr(
        cli_module,
        "device_login",
        lambda **kwargs: {
            "flow": "device_code",
            "token": {"has_access_token": True, "open_id": "ou_test"},
        },
    )
    monkeypatch.setattr(cli_module.FeishuClient, "from_user_context", classmethod(lambda cls, **kwargs: FakeAuthClient()))
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_test"})
    monkeypatch.setattr(
        cli_module,
        "collect_online_personal_inputs",
        lambda **kwargs: {
            "documents": [{"doc_id": "d1", "title": "关键词提取方案", "summary": "上线排期"}],
            "messages": [{"message_id": "m1", "chat_id": "oc_x", "text": "请检查关键词提取上线排期"}],
            "knowledge_items": [{"id": "k1", "title": "关键词提取", "content": "高频主题"}],
            "meta": {"message_count": 1, "document_count": 1},
        },
    )

    code = main(["--output-dir", str(tmp_path), "auth", "login", "--no-browser"])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["profile_bootstrap"]["pending_confirmation"] is True
    assert out["profile_bootstrap"]["profile"]["interests"]
    assert "AI 画像初始化建议" == out["profile_bootstrap"]["card"]["header"]["title"]["content"]


def test_assistant_confirm_profile_cli_marks_confirmation_complete(tmp_path, capsys):
    profile_file = tmp_path / "assistant_profile.json"
    profile_file.write_text(
        json.dumps({"profile": {"persona": "务实推进型", "require_user_confirmation": True}}, ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(["--output-dir", str(tmp_path), "assistant", "confirm-profile", "--profile-file", str(profile_file)])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["confirmed"] is True
    assert out["profile"]["require_user_confirmation"] is False


def test_auth_url_cli_outputs_json(monkeypatch, capsys):
    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")

    code = main(["auth-url", "--scope", "im:chat:read"])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert "client_id=cli_test" in out["authorization_url"]


def test_confused_detect_candidates_cli_outputs_json(tmp_path, capsys):
    messages_file = tmp_path / "messages.json"
    messages_file.write_text(
        json.dumps(
            [
                {"message_id": "m1", "sender": {"id": "u1"}, "content": "先发版再回滚"},
                {"message_id": "m2", "sender": {"id": "u2"}, "parent_id": "m1", "content": "什么？"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    code = main(["confused", "detect-candidates", "--messages-file", str(messages_file)])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["count"] == 1
    assert out["candidates"][0]["target_message_id"] == "m1"


def test_assistant_build_personal_brief_cli_outputs_json(tmp_path, capsys):
    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps(
            [
                {"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"},
                {"doc_id": "d2", "title": "周会纪要", "summary": "例行同步"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    messages_file = tmp_path / "messages.json"
    messages_file.write_text(
        json.dumps(
            [
                {"message_id": "m1", "chat_id": "oc_x", "content": "发布今天截止，尽快确认"},
                {"message_id": "m2", "chat_id": "oc_x", "content": "周会改到明天"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "assistant",
            "build-personal-brief",
            "--documents-file",
            str(documents_file),
            "--messages-file",
            str(messages_file),
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["report"]["summary"]["doc_count"] >= 1


def test_assistant_build_personal_brief_doc_output_is_deprecated(tmp_path, capsys):
    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "assistant",
            "build-personal-brief",
            "--documents-file",
            str(documents_file),
            "--output-format",
            "doc",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["deprecated"]["output_format_doc"] is True


def test_assistant_build_personal_brief_online_uses_collector(monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "FeishuClient", lambda: object())
    monkeypatch.setattr(
        cli_module,
        "collect_online_personal_inputs",
        lambda **kwargs: {
            "documents": [{"doc_id": "d1", "title": "紧急排障", "summary": "线上故障", "url": "", "updated_at": ""}],
            "access_records": [{"doc_id": "d1", "user_id": "ou_test", "action": "view", "count": 2}],
            "messages": [{"message_id": "m1", "chat_id": "oc_x", "text": "故障今天处理", "create_time": ""}],
            "knowledge_items": [{"id": "k1", "title": "故障", "content": "近期高频"}],
            "resolved_target_user_id": "ou_test",
            "meta": {"message_count": 1},
        },
    )

    code = main(["assistant", "build-personal-brief", "--online", "--output-format", "json"])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["meta"]["mode"] == "online"
    assert out["meta"]["resolved_target_user_id"] == "ou_test"
    assert out["report"]["summary"]["doc_count"] == 1


def test_assistant_profile_set_and_get_cli(tmp_path, capsys):
    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "assistant",
            "set-profile",
            "--persona",
            "专业形象",
            "--role",
            "产品经理",
            "--businesses",
            "A增长,B交付",
            "--interests",
            "客户,需求",
            "--mode",
            "hybrid",
            "--weekly-brief-cron",
            "0 10 * * MON",
            "--dual-tower-enabled",
            "--dual-tower-model",
            "text-embedding-3-large",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["profile"]["role"] == "产品经理"

    code = main(["--output-dir", str(tmp_path), "assistant", "get-profile"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["profile"]["persona"] == "专业形象"
    assert out["schedule"]["mode"] == "hybrid"
    assert out["retrieval"]["enabled"] is True
    assert out["retrieval"]["embedding_model"] == "text-embedding-3-large"


def test_assistant_build_personal_brief_uses_profile_file(tmp_path, capsys):
    profile_file = tmp_path / "assistant_profile.json"
    model_file = tmp_path / "dual_tower_model.json"
    model_file.write_text(
        json.dumps(
            {
                "model_type": "dual_tower_baseline_term_weight",
                "token_weights": {"发布": 1.5, "流程": 1.0},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    profile_file.write_text(
        json.dumps(
            {
                "profile": {
                    "persona": "冷静分析",
                    "role": "工程经理",
                    "businesses": ["A增长", "B交付"],
                    "interests": ["上线", "风险"],
                },
                "schedule": {
                    "mode": "scheduled",
                    "weekly_brief_cron": "0 9 * * MON",
                    "nightly_interest_cron": "0 21 * * *",
                },
                "retrieval": {
                    "enabled": True,
                    "embedding_model": "text-embedding-3-large",
                    "model_file": str(model_file),
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "待办：今晚上线"}], ensure_ascii=False),
        encoding="utf-8",
    )
    messages_file = tmp_path / "messages.json"
    messages_file.write_text(
        json.dumps([{"message_id": "m1", "chat_id": "oc_x", "content": "线上风险今晚处理"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "assistant",
            "build-personal-brief",
            "--documents-file",
            str(documents_file),
            "--messages-file",
            str(messages_file),
            "--profile-file",
            str(profile_file),
            "--output-format",
            "all",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["report"]["profile"]["role"] == "工程经理"
    assert out["report"]["interest_card"]["header"]["title"]["content"] == "群聊兴趣消息汇总"
    assert out["report"]["runtime_plan"]["cron_jobs"][0]["job_name"] == "assistant_weekly_brief"
    assert out["report"]["retrieval_plan"]["strategy"] == "dual_tower_trained"
    assert out["report"]["retrieval_plan"]["model_file"] == str(model_file)


def test_assistant_export_dual_tower_samples_cli(tmp_path, capsys):
    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps(
            [
                {"doc_id": "d1", "title": "Release Flow", "summary": "Rollback checklist"},
                {"doc_id": "d2", "title": "Weekly Notes", "summary": "Routine sync"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    access_file = tmp_path / "access.json"
    access_file.write_text(
        json.dumps([{"doc_id": "d1", "user_id": "ou_1", "action": "view", "count": 2}], ensure_ascii=False),
        encoding="utf-8",
    )
    messages_file = tmp_path / "messages.json"
    messages_file.write_text(
        json.dumps([{"message_id": "m1", "chat_id": "oc_x", "content": "release risk rollback review"}], ensure_ascii=False),
        encoding="utf-8",
    )
    output_file = tmp_path / "dual_tower_samples.jsonl"

    code = main(
        [
            "assistant",
            "export-dual-tower-samples",
            "--documents-file",
            str(documents_file),
            "--access-records-file",
            str(access_file),
            "--messages-file",
            str(messages_file),
            "--target-user-id",
            "ou_1",
            "--role",
            "PM",
            "--interests",
            "release,risk",
            "--output-file",
            str(output_file),
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["sample_count"] == 1
    assert output_file.exists()


def test_assistant_train_dual_tower_cli(tmp_path, capsys):
    samples_file = tmp_path / "samples.jsonl"
    samples_file.write_text(
        json.dumps(
            {
                "user_id": "ou_1",
                "doc_id": "d1",
                "label": 1,
                "strength": 2,
                "user_tower_text": "role: PM | interests: release, risk",
                "positive_doc_text": "title: Release Flow | summary: Rollback checklist | business: Release | doc_type: file",
                "negative_doc_texts": [
                    "title: Weekly Notes | summary: Routine sync | business: General | doc_type: file"
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_file = tmp_path / "dual_tower_model.json"

    code = main(
        [
            "assistant",
            "train-dual-tower",
            "--samples-file",
            str(samples_file),
            "--output-file",
            str(output_file),
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["model_type"] == "dual_tower_baseline_term_weight"
    assert out["quality"]["evaluated_samples"] == 1
    assert output_file.exists()


def test_assistant_recommend_falls_back_to_openclaw_when_samples_insufficient(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_module, "FeishuClient", lambda: object())
    monkeypatch.setattr(
        cli_module,
        "collect_online_personal_inputs",
        lambda **kwargs: {
            "documents": [{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布", "url": "", "updated_at": ""}],
            "access_records": [{"doc_id": "d1", "user_id": "ou_test", "action": "view", "count": 1}],
            "messages": [{"message_id": "m1", "chat_id": "oc_x", "text": "发布今天截止", "create_time": ""}],
            "knowledge_items": [],
            "resolved_target_user_id": "ou_test",
            "meta": {"message_count": 1},
        },
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "assistant",
            "recommend",
            "--role",
            "PM",
            "--interests",
            "发布,风险",
            "--dual-tower-min-samples",
            "2",
            "--output-format",
            "json",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["report"]["retrieval_plan"]["strategy"] == "openclaw_fallback"
    assert out["meta"]["auto_retrieval"]["enough_data"] is False
    assert out["meta"]["auto_retrieval"]["accumulated_sample_count"] == 1


def test_assistant_recommend_trains_and_uses_dual_tower_when_samples_sufficient(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_module, "FeishuClient", lambda: object())
    monkeypatch.setattr(
        cli_module,
        "collect_online_personal_inputs",
        lambda **kwargs: {
            "documents": [
                {"doc_id": "d1", "title": "发布流程", "summary": "审批后发布", "url": "", "updated_at": ""},
                {"doc_id": "d2", "title": "周会纪要", "summary": "例行同步", "url": "", "updated_at": ""},
            ],
            "access_records": [{"doc_id": "d1", "user_id": "ou_test", "action": "view", "count": 2}],
            "messages": [{"message_id": "m1", "chat_id": "oc_x", "text": "发布风险今晚处理", "create_time": ""}],
            "knowledge_items": [],
            "resolved_target_user_id": "ou_test",
            "meta": {"message_count": 1},
        },
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "assistant",
            "recommend",
            "--role",
            "PM",
            "--interests",
            "发布,风险",
            "--dual-tower-min-samples",
            "1",
            "--output-format",
            "json",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["report"]["retrieval_plan"]["strategy"] == "dual_tower_trained"
    assert out["meta"]["auto_retrieval"]["enough_data"] is True
    assert (tmp_path / "assistant_dual_tower_model.json").exists()


def test_assistant_recommend_push_uses_bot_client_and_prompts_profile_after_cards(monkeypatch, tmp_path, capsys):
    send_order: list[str] = []

    class FakeUserClient:
        pass

    class FakeBotClient:
        def send_message(self, receive_id, msg_type, content, receive_id_type="chat_id"):
            title = content.get("header", {}).get("title", {}).get("content", "")
            send_order.append(title)
            return {"message_id": f"msg_{len(send_order)}", "chat_id": "oc_push", "msg_type": msg_type}

    monkeypatch.setattr(cli_module, "build_user_feishu_client", lambda args, require_token=False: FakeUserClient())
    monkeypatch.setattr(cli_module, "build_bot_feishu_client", lambda: FakeBotClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_test"})
    monkeypatch.setattr(
        cli_module,
        "collect_online_personal_inputs",
        lambda **kwargs: {
            "documents": [{"doc_id": "d1", "title": "发布流程", "summary": "今晚发布", "url": "", "updated_at": ""}],
            "access_records": [],
            "messages": [{"message_id": "m1", "chat_id": "oc_x", "text": "发布今天截止", "create_time": ""}],
            "knowledge_items": [],
            "resolved_target_user_id": "ou_test",
            "meta": {"message_count": 1, "document_count": 1},
        },
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "assistant",
            "recommend",
            "--push",
            "--output-format",
            "json",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["meta"]["push"]["profile_setup_prompted"] is True
    assert send_order[:2] == ["个人助理聚合建议", "群聊兴趣消息汇总"]
    assert send_order[2] == "AI 画像初始化建议"


def test_assistant_push_defaults_to_personal_open_id(monkeypatch, tmp_path, capsys):
    calls: list[dict[str, str]] = []

    class FakeClient:
        def send_message(self, receive_id, msg_type, content, receive_id_type="chat_id"):
            calls.append(
                {
                    "receive_id": receive_id,
                    "receive_id_type": receive_id_type,
                    "msg_type": msg_type,
                }
            )
            return {"message_id": "om_test", "chat_id": "oc_personal", "msg_type": msg_type}

    monkeypatch.setattr(cli_module, "FeishuClient", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "assistant",
            "build-personal-brief",
            "--documents-file",
            str(documents_file),
            "--push",
            "--output-format",
            "json",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert calls
    assert calls[0]["receive_id_type"] == "open_id"
    assert calls[0]["receive_id"] == "ou_self"
    assert out["meta"]["push"]["receive_id_type"] == "open_id"
    assert len(calls) == 3
    assert out["meta"]["push"]["profile_setup_prompted"] is True
    assert out["meta"]["push"]["summary_enabled"] is True
    assert out["meta"]["push"]["interest_enabled"] is True
    assert out["meta"]["push"]["doc_push_enabled"] is False


def test_assistant_push_explicit_chat_id_overrides_personal(monkeypatch, tmp_path, capsys):
    calls: list[dict[str, str]] = []

    class FakeClient:
        def send_message(self, receive_id, msg_type, content, receive_id_type="chat_id"):
            calls.append(
                {
                    "receive_id": receive_id,
                    "receive_id_type": receive_id_type,
                    "msg_type": msg_type,
                }
            )
            return {"message_id": "om_test2", "chat_id": receive_id, "msg_type": msg_type}

    monkeypatch.setattr(cli_module, "FeishuClient", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "assistant",
            "build-personal-brief",
            "--documents-file",
            str(documents_file),
            "--push",
            "--receive-chat-id",
            "oc_group_x",
            "--output-format",
            "json",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert calls
    assert calls[0]["receive_id_type"] == "chat_id"
    assert calls[0]["receive_id"] == "oc_group_x"
    assert out["meta"]["push"]["receive_id_type"] == "chat_id"


def test_assistant_push_card_also_pushes_interest_card(monkeypatch, tmp_path, capsys):
    calls: list[dict[str, str]] = []

    class FakeClient:
        def send_message(self, receive_id, msg_type, content, receive_id_type="chat_id"):
            calls.append({"receive_id": receive_id, "receive_id_type": receive_id_type, "msg_type": msg_type})
            return {"message_id": f"om_{len(calls)}", "chat_id": receive_id, "msg_type": msg_type}

    monkeypatch.setattr(cli_module, "FeishuClient", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "assistant",
            "build-personal-brief",
            "--documents-file",
            str(documents_file),
            "--push",
            "--output-format",
            "card",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert len(calls) == 3
    assert out["meta"]["push"]["profile_setup_prompted"] is True
    assert out["meta"]["push"]["interest_enabled"] is True


def test_assistant_push_both_cards_requested_keeps_single_card(monkeypatch, tmp_path, capsys):
    calls: list[dict[str, str]] = []

    class FakeClient:
        def send_message(self, receive_id, msg_type, content, receive_id_type="chat_id"):
            calls.append({"receive_id": receive_id, "receive_id_type": receive_id_type, "msg_type": msg_type})
            return {"message_id": f"om_{len(calls)}", "chat_id": receive_id, "msg_type": msg_type}

    monkeypatch.setattr(cli_module, "FeishuClient", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "assistant",
            "build-personal-brief",
            "--documents-file",
            str(documents_file),
            "--push",
            "--push-summary-card",
            "--push-interest-card",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert len(calls) == 3
    assert out["meta"]["push"]["profile_setup_prompted"] is True
    assert out["meta"]["push"]["summary_enabled"] is True
    assert out["meta"]["push"]["interest_enabled"] is True


def test_assistant_push_interest_card_failure_does_not_fail_command(monkeypatch, tmp_path, capsys):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def send_message(self, receive_id, msg_type, content, receive_id_type="chat_id"):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interest card push failed")
            return {"message_id": f"om_{self.calls}", "chat_id": receive_id, "msg_type": msg_type}

    monkeypatch.setattr(cli_module, "FeishuClient", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "assistant",
            "build-personal-brief",
            "--documents-file",
            str(documents_file),
            "--push",
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["meta"]["push"]["errors"]
    assert out["meta"]["push"]["errors"][0]["card"] == "interest"


def test_lingo_upsert_list_delete_cli(tmp_path, capsys):
    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "lingo",
            "upsert",
            "--no-remote",
            "--keyword",
            "北极星指标",
            "--type",
            "black",
            "--value",
            "团队核心牵引指标",
            "--aliases",
            "北极星,North Star",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["entry"]["keyword"] == "北极星指标"

    code = main(["--output-dir", str(tmp_path), "lingo", "list"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["count"] == 1
    assert out["entries"][0]["keyword"] == "北极星指标"

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "lingo",
            "delete",
            "--no-remote",
            "--keyword",
            "北极星指标",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["local"]["deleted"] is True


def test_lingo_sync_from_file_cli(tmp_path, capsys):
    input_file = tmp_path / "judgements.json"
    input_file.write_text(
        json.dumps(
            [
                {"keyword": "A/B实验", "type": "black", "value": "用于对照验证的实验方法"},
                {"keyword": "你好", "type": "nothing", "value": ""},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "lingo",
            "sync-from-file",
            "--no-remote",
            "--input-file",
            str(input_file),
            "--publishable-only",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["count"] == 1
    assert out["entries"][0]["keyword"] == "A/B实验"
