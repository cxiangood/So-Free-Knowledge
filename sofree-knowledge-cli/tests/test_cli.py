import json

from sofree_knowledge.cli import main


def test_set_knowledge_scope_cli_outputs_json(tmp_path, capsys):
    code = main([
        "--output-dir",
        str(tmp_path),
        "set-knowledge-scope",
        "oc_test",
        "chat_only",
    ])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["scope"] == "chat_only"


def test_get_knowledge_scope_cli_outputs_json(tmp_path, capsys):
    main(["--output-dir", str(tmp_path), "set-knowledge-scope", "oc_test", "chat_only"])
    capsys.readouterr()

    code = main(["--output-dir", str(tmp_path), "get-knowledge-scope", "oc_test"])

    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["scope"] == "chat_only"
