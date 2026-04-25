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
