import json

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
