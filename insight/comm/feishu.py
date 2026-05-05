from __future__ import annotations

from pathlib import Path

from utils import getenv_required, load_env_file as _load_env_file


def resolve_sender_credentials() -> tuple[str, str]:
    return getenv_required("APP_ID"), getenv_required("SECRET_ID")


def load_env_file(path: str = "") -> None:
    env_path = Path(path).expanduser() if path.strip() else Path.cwd() / ".env"
    _load_env_file(env_path)


__all__ = ["load_env_file", "resolve_sender_credentials"]

