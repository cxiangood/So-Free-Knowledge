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


def test_assistant_build_personal_brief_uses_profile_file(tmp_path, capsys):
    profile_file = tmp_path / "assistant_profile.json"
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
    assert len(calls) == 2
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
    assert len(calls) == 2
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
    assert len(calls) == 2
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
