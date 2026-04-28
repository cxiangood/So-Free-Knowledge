from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str = "") -> None:
    env_path = Path(path).expanduser() if path.strip() else Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_sender_credentials() -> tuple[str, str]:
    app_id = (os.getenv("CARD_SENDER_APP_ID", "") or os.getenv("FEISHU_APP_ID", "")).strip()
    app_secret = (os.getenv("CARD_SENDER_APP_SECRET", "") or os.getenv("FEISHU_APP_SECRET", "")).strip()
    return app_id, app_secret


__all__ = ["load_env_file", "resolve_sender_credentials"]

