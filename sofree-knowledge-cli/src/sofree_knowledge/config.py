from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from utils import getenv as _getenv
    from utils import load_env_file as _shared_load_env_file
except ModuleNotFoundError:  # pragma: no cover
    _getenv = os.getenv
    _shared_load_env_file = None


DEFAULT_TOKEN_FILE = Path.home() / ".feishu" / "token.json"


def load_env_file(path: str | Path | None) -> None:
    if _shared_load_env_file is not None:
        _shared_load_env_file(path, override=False)
        return
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def first_env(*names: str) -> str | None:
    for name in names:
        value = _getenv(name)
        if value:
            return value
    return None


def get_app_credentials() -> tuple[str | None, str | None]:
    return (
        first_env("FEISHU_APP_ID", "SOFREE_FEISHU_APP_ID", "LARKSUITE_CLI_APP_ID", "APP_ID"),
        first_env("FEISHU_APP_SECRET", "SOFREE_FEISHU_APP_SECRET", "LARKSUITE_CLI_APP_SECRET", "APP_SECRET"),
    )


def resolve_env_file(path: str | Path | None, output_dir: str | Path = ".") -> str:
    if path:
        return str(Path(path).expanduser())
    root = Path(output_dir).expanduser()
    cwd = Path.cwd()
    candidates = [
        root / ".env",
        cwd / ".env",
        cwd.parent / ".env",
        root / "So-Free-Knowledge" / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def get_user_access_token(token_file: str | Path | None = None) -> str | None:
    env_token = _getenv("FEISHU_ACCESS_TOKEN")
    if env_token:
        return env_token
    path = Path(token_file).expanduser() if token_file else DEFAULT_TOKEN_FILE
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8-sig")
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'"access_token"\s*:\s*"([^"]+)"', raw)
        return match.group(1) if match else None
    if isinstance(data, dict):
        token = data.get("access_token")
        return str(token) if token else None
    return None


def get_user_identity(token_file: str | Path | None = None) -> dict[str, str]:
    path = Path(token_file).expanduser() if token_file else DEFAULT_TOKEN_FILE
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8-sig")
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        result: dict[str, str] = {}
        for key in ("open_id", "user_id", "union_id"):
            match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', raw)
            if match:
                result[key] = match.group(1)
        return result
    if not isinstance(data, dict):
        return {}
    result = {}
    for key in ("open_id", "user_id", "union_id"):
        value = data.get(key)
        if value:
            result[key] = str(value)
    return result
