import json

from pathlib import Path

import sofree_knowledge.cli as cli_module
from sofree_knowledge.assistant.profile import load_assistant_profile_config
from sofree_knowledge.cli import main
from sofree_knowledge.lingo_auto import build_lingo_openclaw_prompt


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


def test_cli_logs_to_file_without_polluting_stdout(tmp_path, capsys):
    token_file = tmp_path / "missing.json"
    log_file = tmp_path / "logs" / "cli.log"

    code = main(
        [
            "--log-level",
            "DEBUG",
            "--log-file",
            str(log_file),
            "--quiet",
            "auth-status",
            "--token-file",
            str(token_file),
        ]
    )

    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert code == 0
    assert out["ok"] is True
    assert captured.err == ""
    assert log_file.exists()
    assert "running command: auth-status" in log_file.read_text(encoding="utf-8")


def test_cli_quiet_failure_keeps_stderr_machine_readable(tmp_path, capsys):
    missing_file = tmp_path / "missing.txt"

    code = main(["--quiet", "confused", "parse-judgement", "--judgement-file", str(missing_file)])

    captured = capsys.readouterr()
    err = json.loads(captured.err)
    assert code == 1
    assert captured.out == ""
    assert err["ok"] is False
    assert "Traceback" not in captured.err


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
    monkeypatch.setenv("APP_ID", "cli_test")

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


def test_brief_shortcut_pushes_card_with_defaults(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def fake_recommend_command_result(args, **kwargs):
        captured["args"] = args
        return {
            "ok": True,
            "meta": {
                "push": {
                    "enabled": args.push,
                    "interest_enabled": args.push_interest_card,
                    "summary_enabled": args.push_summary_card,
                }
            },
            "output_format": args.output_format,
        }

    monkeypatch.setattr(cli_module, "recommend_command_result", fake_recommend_command_result)

    code = main(["brief"])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["output_format"] == "card"
    assert out["meta"]["push"]["enabled"] is True
    assert out["meta"]["push"]["interest_enabled"] is True
    assert out["meta"]["push"]["summary_enabled"] is False
    args = captured["args"]
    assert args.command == "brief"
    assert args.recent_days == 7
    assert args.max_chats == 20


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


def test_assistant_profile_uses_user_scoped_output_dir(tmp_path, capsys):
    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--user-open-id",
            "ou_scope_user",
            "assistant",
            "set-profile",
            "--persona",
            "专业形象",
            "--role",
            "产品经理",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert "users" in out["profile_file"]
    assert (tmp_path / "users" / "ou_scope_user" / "assistant_profile.json").exists()


def test_assistant_profile_auto_uses_user_scope_from_token(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_auto_scope"})

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "assistant",
            "set-profile",
            "--persona",
            "自动隔离画像",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert (tmp_path / "users" / "ou_auto_scope" / "assistant_profile.json").exists()


def test_scoped_profile_loads_from_legacy_root_file(tmp_path):
    legacy_profile = tmp_path / "assistant_profile.json"
    legacy_profile.write_text(
        json.dumps({"profile": {"persona": "旧画像", "interests": ["飞书"]}}, ensure_ascii=False),
        encoding="utf-8",
    )

    parsed = load_assistant_profile_config(
        output_dir=str(tmp_path / "users" / "ou_legacy"),
        profile_file="",
    )

    assert parsed["profile"]["persona"] == "旧画像"


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
    assert Path(out["meta"]["auto_retrieval"]["model_file"]).exists()


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
    assert out["meta"]["push"]["recommendation_deferred_until_profile_confirmed"] is True
    assert out["report_deferred"] is True
    assert "profile_setup_card" in out
    assert send_order == ["AI 画像初始化建议"]


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

    monkeypatch.setattr(cli_module, "build_bot_feishu_client", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
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
    assert len(calls) == 1
    assert out["meta"]["push"]["profile_setup_prompted"] is True
    assert out["report_deferred"] is True
    assert out["meta"]["push"]["summary_enabled"] is False
    assert out["meta"]["push"]["interest_enabled"] is False
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

    monkeypatch.setattr(cli_module, "build_bot_feishu_client", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
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

    monkeypatch.setattr(cli_module, "build_bot_feishu_client", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
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
    assert len(calls) == 1
    assert out["meta"]["push"]["profile_setup_prompted"] is True
    assert out["meta"]["push"]["interest_enabled"] is False


def test_assistant_push_both_cards_requested_keeps_single_card(monkeypatch, tmp_path, capsys):
    calls: list[dict[str, str]] = []

    class FakeClient:
        def send_message(self, receive_id, msg_type, content, receive_id_type="chat_id"):
            calls.append({"receive_id": receive_id, "receive_id_type": receive_id_type, "msg_type": msg_type})
            return {"message_id": f"om_{len(calls)}", "chat_id": receive_id, "msg_type": msg_type}

    monkeypatch.setattr(cli_module, "build_bot_feishu_client", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
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
    assert len(calls) == 1
    assert out["meta"]["push"]["profile_setup_prompted"] is True
    assert out["meta"]["push"]["summary_enabled"] is False
    assert out["meta"]["push"]["interest_enabled"] is False


def test_assistant_push_profile_setup_failure_does_not_fail_command(monkeypatch, tmp_path, capsys):
    class FakeClient:
        def send_message(self, receive_id, msg_type, content, receive_id_type="chat_id"):
            raise RuntimeError("profile setup push failed")

    monkeypatch.setattr(cli_module, "build_bot_feishu_client", lambda: FakeClient())
    monkeypatch.setattr(cli_module, "get_user_identity", lambda token_file=None: {"open_id": "ou_self"})

    documents_file = tmp_path / "documents.json"
    documents_file.write_text(
        json.dumps([{"doc_id": "d1", "title": "发布流程", "summary": "审批后发布"}], ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
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
    assert out["meta"]["push"]["errors"][0]["card"] == "profile_setup"


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


def test_lingo_sync_from_file_preserves_aliases(tmp_path, capsys):
    input_file = tmp_path / "judgements_with_aliases.json"
    input_file.write_text(
        json.dumps(
            [
                {
                    "keyword": "JEPA",
                    "type": "black",
                    "value": "团队内部使用的模型项目简称",
                    "aliases": ["Joint Embedding", "jepa"],
                }
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
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["entries"][0]["aliases"] == ["Joint Embedding", "jepa"]
    assert out["entries"][0]["entry"]["aliases"] == ["Joint Embedding", "jepa"]


def test_lingo_auto_sync_cli_emits_ai_review_prompt_without_sync(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "run_lingo_auto_pipeline",
        lambda **kwargs: {
            "ok": True,
            "skipped": False,
            "run_id": "20260503T120000Z",
            "candidate_count": 2,
            "candidates": [
                {"keyword": "JEPA", "frequency": 5, "context_count": 2},
                {"keyword": "今天", "frequency": 5, "context_count": 1},
            ],
            "ai_review": {"prompt": "review this"},
        },
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "lingo",
            "auto-sync",
            "--no-remote",
            "--publishable-only",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["candidate_count"] == 2
    assert out["sync"]["performed"] is False
    assert out["ai_review"]["prompt"] == "review this"


def test_lingo_write_shortcut_reuses_auto_sync_defaults(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def fake_run_lingo_auto_pipeline(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "ok": True,
            "skipped": False,
            "run_id": "20260505T120000Z",
            "candidate_count": 1,
            "candidates": [{"keyword": "JEPA", "frequency": 3, "context_count": 2}],
            "ai_review": {"prompt": "review this"},
        }

    monkeypatch.setattr(cli_module, "run_lingo_auto_pipeline", fake_run_lingo_auto_pipeline)

    code = main(["lingo-write", "--no-remote"])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["sync"]["performed"] is False
    assert out["candidate_count"] == 1
    kwargs = captured["kwargs"]
    assert kwargs["recent_days"] == 7
    assert kwargs["max_chats"] == 200
    assert kwargs["top_keywords"] == 30
    assert kwargs["candidate_limit"] == 20


def test_lingo_auto_sync_cli_applies_ai_review_judgements_and_appends_sense(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "run_lingo_auto_pipeline",
        lambda **kwargs: {
            "ok": True,
            "skipped": False,
            "run_id": "20260503T120000Z",
            "candidate_count": 1,
            "candidates": [{"keyword": "JEPA", "frequency": 5, "context_count": 2}],
            "ai_review": {"prompt": "review this"},
        },
    )
    scoped_root = tmp_path / "users" / "ou_lingo_test"
    scoped_root.mkdir(parents=True, exist_ok=True)
    existing_file = scoped_root / "lingo_entries.json"
    existing_file.write_text(
        json.dumps(
            {
                "entries": {
                    "JEPA": {
                        "keyword": "JEPA",
                        "type": "black",
                        "value": "旧释义",
                        "aliases": ["old"],
                        "senses": [
                            {
                                "sense_id": "sense_old",
                                "type": "black",
                                "value": "旧释义",
                                "aliases": ["old"],
                                "entity_id": "",
                                "context_ids": ["ctx_old"],
                            }
                        ],
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    judgements_file = tmp_path / "ai_review_judgements.json"
    judgements_file.write_text(
        json.dumps(
            [
                {
                    "keyword": "JEPA",
                    "decision": "append_new_sense",
                    "type": "black",
                    "value": "团队内部使用的模型项目简称",
                    "context_ids": ["ctx_1"],
                    "matched_existing_sense_ids": ["sense_old"],
                    "aliases": ["Joint Embedding"],
                    "reason": "与旧释义不同",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--user-open-id",
            "ou_lingo_test",
            "lingo",
            "auto-sync",
            "--no-remote",
            "--judgements-file",
            str(judgements_file),
            "--publishable-only",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["sync"]["performed"] is True
    assert out["sync"]["count"] == 1
    entry = out["sync"]["entries"][0]["entry"]
    assert entry["keyword"] == "JEPA"
    assert len(entry["senses"]) == 2
    assert entry["senses"][0]["value"] == "旧释义"
    assert entry["senses"][1]["value"] == "团队内部使用的模型项目简称"


def test_lingo_sync_from_file_skips_remote_create_when_matching_sense_already_has_entity_id(
    tmp_path,
    monkeypatch,
    capsys,
):
    scoped_root = tmp_path / "users" / "ou_lingo_test"
    scoped_root.mkdir(parents=True, exist_ok=True)
    existing_file = scoped_root / "lingo_entries.json"
    existing_file.write_text(
        json.dumps(
            {
                "entries": {
                    "JEPA": {
                        "keyword": "JEPA",
                        "type": "black",
                        "value": "旧释义",
                        "entity_id": "",
                        "senses": [
                            {
                                "sense_id": "sense_old",
                                "type": "black",
                                "value": "旧释义",
                                "entity_id": "",
                                "context_ids": ["ctx_old"],
                            },
                            {
                                "sense_id": "sense_new",
                                "type": "black",
                                "value": "团队内部使用的模型项目简称",
                                "entity_id": "ent_existing",
                                "context_ids": ["ctx_new"],
                            },
                        ],
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    judgements_file = tmp_path / "judgements.json"
    judgements_file.write_text(
        json.dumps(
            [
                {
                    "keyword": "JEPA",
                    "type": "black",
                    "value": "团队内部使用的模型项目简称",
                    "aliases": ["Joint Embedding"],
                    "context_ids": ["ctx_newer"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class FailIfCalledClient:
        def create_lingo_entity(self, **kwargs):
            raise AssertionError("remote create should be skipped for already synced matching sense")

    monkeypatch.setattr(cli_module, "_instantiate_feishu_client", lambda: FailIfCalledClient())

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--user-open-id",
            "ou_lingo_test",
            "lingo",
            "sync-from-file",
            "--input-file",
            str(judgements_file),
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["entries"][0]["remote_create_skipped"] is True
    assert out["entries"][0]["entity_id"] == "ent_existing"


def test_lingo_auto_sync_skips_remote_create_when_matching_sense_already_has_entity_id(
    tmp_path,
    monkeypatch,
):
    scoped_root = tmp_path / "users" / "ou_lingo_test"
    scoped_root.mkdir(parents=True, exist_ok=True)
    (scoped_root / "lingo_entries.json").write_text(
        json.dumps(
            {
                "entries": {
                    "JEPA": {
                        "keyword": "JEPA",
                        "type": "black",
                        "value": "旧释义",
                        "entity_id": "",
                        "senses": [
                            {
                                "sense_id": "sense_old",
                                "type": "black",
                                "value": "旧释义",
                                "entity_id": "",
                            },
                            {
                                "sense_id": "sense_new",
                                "type": "black",
                                "value": "团队内部使用的模型项目简称",
                                "entity_id": "ent_existing",
                            },
                        ],
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    judgements = [
        {
            "keyword": "JEPA",
            "decision": "append_new_sense",
            "type": "black",
            "value": "团队内部使用的模型项目简称",
            "context_ids": ["ctx_1"],
            "matched_existing_sense_ids": ["sense_new"],
            "aliases": ["Joint Embedding"],
        }
    ]

    class FailIfCalledClient:
        def create_lingo_entity(self, **kwargs):
            raise AssertionError("remote create should be skipped for already synced matching sense")

    result = cli_module.sync_ai_review_judgements(
        output_dir=scoped_root,
        judgements=judgements,
        source="lingo_auto",
        remote=True,
        write_local=True,
        force_remote_create=False,
        client=FailIfCalledClient(),
        publishable_only=True,
    )

    assert result["entries"][0]["remote_create_skipped"] is True
    assert result["entries"][0]["entity_id"] == "ent_existing"


def test_lingo_auto_sync_cli_keeps_web_search_pending_without_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "run_lingo_auto_pipeline",
        lambda **kwargs: {
            "ok": True,
            "skipped": False,
            "run_id": "20260504T010000Z",
            "candidate_count": 1,
            "candidates": [{"keyword": "llm4rec", "frequency": 4, "context_count": 2}],
            "ai_review": {"prompt": "review this"},
        },
    )
    scoped_root = tmp_path / "users" / "ou_pending_test"
    scoped_root.mkdir(parents=True, exist_ok=True)
    judgements_file = tmp_path / "ai_review_pending.json"
    judgements_file.write_text(
        json.dumps(
            [
                {
                    "keyword": "llm4rec",
                    "decision": "create_entry",
                    "type": "black",
                    "value": "团队讨论中的模型名",
                    "refined_value": "一个用于推荐场景的大语言模型方法名",
                    "context_ids": ["ctx_1"],
                    "matched_existing_sense_ids": [],
                    "aliases": [],
                    "web_search_needed": True,
                    "search_queries": ["llm4rec recommendation model", "llm4rec paper"],
                    "search_goal": "确认该模型名的标准释义",
                    "reason": "需要外部资料校正定义",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--user-open-id",
            "ou_pending_test",
            "lingo",
            "auto-sync",
            "--no-remote",
            "--judgements-file",
            str(judgements_file),
            "--publishable-only",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["sync"]["performed"] is True
    assert out["sync"]["count"] == 0
    assert out["sync"]["pending_web_search_count"] == 1
    assert out["sync"]["pending_web_search"][0]["keyword"] == "llm4rec"
    assert not (scoped_root / "lingo_entries.json").exists()


def test_lingo_auto_review_prompt_contains_chinese_noun_positive_example():
    prompt = build_lingo_openclaw_prompt(
        [
            {
                "keyword": "神经网络",
                "frequency": 3,
                "context_count": 2,
                "context_ids": ["ctx_1"],
                "semantic_density": 0.0,
                "attention_entropy": 0.0,
                "bert_score": 0.0,
                "initial_type": "key",
                "initial_value": "机器学习中的一种模型结构",
                "initial_ratio": 0.8,
                "existing_entry": None,
                "related_existing_entries": [],
                "contexts": [{"context_id": "ctx_1", "text": "这里讨论神经网络模型结构。"}],
            }
        ],
        chunk_id=1,
        total_chunks=1,
        bert_effective=False,
    )

    assert "神经网络" in prompt
    assert "稳定的技术/行业名词" in prompt
    assert "LLM" in prompt
    assert "GitHub" in prompt
    assert "TPM" in prompt
    assert "你就是这批候选词的最终判断者" in prompt
    assert "initial_type / initial_value" in prompt
    assert "不要要求用户去配置 ~/.sofree/knowledge_config.json" in prompt
    assert "--force-remote-create" in prompt


def test_lingo_candidate_keyword_filter_rejects_system_template_fields():
    assert cli_module is not None
    from sofree_knowledge.lingo_auto import (
        _is_candidate_keyword,
        _normalize_initial_sense,
        _sanitize_classifier_result_for_external_review,
    )

    assert _is_candidate_keyword("from_user") is False
    assert _is_candidate_keyword("to_chatters") is False
    assert _is_candidate_keyword("template") is False
    assert _is_candidate_keyword("divider_text") is False
    assert _normalize_initial_sense({"type": "confused", "sense": "未配置LLM分类器"}) == {
        "type": "nothing",
        "sense": "",
        "ratio": 0.0,
    }
    sanitized = _sanitize_classifier_result_for_external_review(
        {
            "group_classification": {"group-0": {"TPC": {"type": "confused", "sense": "未配置LLM分类器"}}},
            "classification_results": {"TPC": [{"type": "confused", "sense": "未配置LLM分类器", "ratio": 1.0}]},
        }
    )
    assert sanitized["local_classifier_enabled"] is False
    assert sanitized["review_mode"] == "external_ai_only"
    assert sanitized["group_classification"]["group-0"]["TPC"] == {"type": "nothing", "sense": ""}
    assert sanitized["classification_results"]["TPC"] == [{"type": "nothing", "sense": "", "ratio": 0.0}]


def test_wikisheet_create_sheet_routed_from_main_cli(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module.wikisheet_module,
        "cmd_create_sheet",
        lambda args: {
            "ok": True,
            "action": "create-sheet",
            "title": args.title,
            "resolved_space_id": "space_xxx",
            "spreadsheet_token": "sht_xxx",
        },
    )

    code = main(["wikisheet", "create-sheet", "--title", "经营周报", "--space-id", "my_library"])
    out = json.loads(capsys.readouterr().out)

    assert code == 0
    assert out["ok"] is True
    assert out["action"] == "create-sheet"
    assert out["title"] == "经营周报"
