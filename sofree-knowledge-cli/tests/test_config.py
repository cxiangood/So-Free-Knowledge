from pathlib import Path

from sofree_knowledge.config import get_app_credentials, resolve_env_file


def test_get_app_credentials_supports_app_id_aliases(monkeypatch):
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
    monkeypatch.setenv("APP_ID", "cli_alias")
    monkeypatch.setenv("APP_SECRET", "secret_alias")

    app_id, app_secret = get_app_credentials()

    assert app_id == "cli_alias"
    assert app_secret == "secret_alias"


def test_resolve_env_file_prefers_output_dir(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = output_dir / ".env"
    expected.write_text("FEISHU_APP_ID=cli_test\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved = resolve_env_file(None, output_dir=output_dir)

    assert Path(resolved) == expected


def test_resolve_env_file_falls_back_to_parent_cwd(tmp_path, monkeypatch):
    workdir = tmp_path / "repo" / "sofree-knowledge-cli"
    workdir.mkdir(parents=True, exist_ok=True)
    parent_env = workdir.parent / ".env"
    parent_env.write_text("FEISHU_APP_ID=cli_parent\n", encoding="utf-8")
    monkeypatch.chdir(workdir)

    resolved = resolve_env_file(None, output_dir=".")

    assert Path(resolved) == parent_env
