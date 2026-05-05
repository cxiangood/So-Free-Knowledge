from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.yaml"


def _resolve_path(path: str | Path | None = None) -> Path:
    if path is None or str(path).strip() == "":
        return _DEFAULT_CONFIG_PATH
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    return config_path


@lru_cache(maxsize=16)
def _load_config_cached(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return raw


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    return dict(_load_config_cached(str(_resolve_path(path))))


def reload_config() -> None:
    _load_config_cached.cache_clear()


def get_config_value(key: str, default: Any = None, *, path: str | Path | None = None) -> Any:
    current: Any = _load_config_cached(str(_resolve_path(path)))
    for part in str(key or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def get_config_section(key: str, *, path: str | Path | None = None) -> dict[str, Any]:
    value = get_config_value(key, {}, path=path)
    return value if isinstance(value, dict) else {}


def get_config_str(key: str, default: str = "", *, path: str | Path | None = None) -> str:
    value = get_config_value(key, default, path=path)
    return str(value if value is not None else default)


def get_config_int(key: str, default: int = 0, *, path: str | Path | None = None) -> int:
    value = get_config_value(key, default, path=path)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def get_config_float(key: str, default: float = 0.0, *, path: str | Path | None = None) -> float:
    value = get_config_value(key, default, path=path)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_config_bool(key: str, default: bool = False, *, path: str | Path | None = None) -> bool:
    value = get_config_value(key, default, path=path)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def get_config_path(key: str, default: str | Path = "", *, path: str | Path | None = None) -> Path:
    value = get_config_value(key, default, path=path)
    config_path = Path(str(value if value is not None else default)).expanduser()
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    return config_path


__all__ = [
    "load_config",
    "reload_config",
    "get_config_value",
    "get_config_section",
    "get_config_str",
    "get_config_int",
    "get_config_float",
    "get_config_bool",
    "get_config_path",
]
