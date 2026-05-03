from __future__ import annotations

from pathlib import Path

from utils import getenv, load_env_file as _load_env_file


def resolve_sender_credentials() -> tuple[str, str]:
    app_id = (getenv("CARD_SENDER_APP_ID", "") or getenv("FEISHU_APP_ID", "")).strip()
    app_secret = (getenv("CARD_SENDER_APP_SECRET", "") or getenv("FEISHU_APP_SECRET", "")).strip()
    return app_id, app_secret


def load_env_file(path: str = "") -> None:
    env_path = Path(path).expanduser() if path.strip() else Path.cwd() / ".env"
    _load_env_file(env_path)


__all__ = ["load_env_file", "resolve_sender_credentials"]

