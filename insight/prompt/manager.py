from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class PromptNotFoundError(KeyError):
    """Raised when a prompt key does not exist."""


@dataclass(slots=True)
class PromptStore:
    """YAML-backed prompt store with lightweight reload-on-change behavior."""

    yaml_path: Path
    _cache: dict[str, str] | None = None
    _mtime_ns: int | None = None

    def reload(self) -> None:
        """Force reload prompts from disk."""
        self._cache = self._load_yaml(self.yaml_path)
        stat = self.yaml_path.stat()
        self._mtime_ns = int(stat.st_mtime_ns)

    def get(self, key: str) -> str:
        """Get a prompt by key, raises PromptNotFoundError when missing."""
        prompts = self._ensure_loaded()
        if key not in prompts:
            raise PromptNotFoundError(f"Prompt key not found: {key}")
        return prompts[key]

    def keys(self) -> list[str]:
        """List all available prompt keys in sorted order."""
        prompts = self._ensure_loaded()
        return sorted(prompts.keys())

    def exists(self, key: str) -> bool:
        """Check whether a prompt key exists."""
        prompts = self._ensure_loaded()
        return key in prompts

    def _ensure_loaded(self) -> dict[str, str]:
        if self._cache is None:
            self.reload()
            return self._cache or {}
        try:
            mtime_ns = int(self.yaml_path.stat().st_mtime_ns)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt YAML file not found: {self.yaml_path}") from None
        if self._mtime_ns != mtime_ns:
            self.reload()
        return self._cache or {}

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, str]:
        if not path.exists():
            raise FileNotFoundError(f"Prompt YAML file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("Prompt YAML root must be a mapping")

        prompts_raw: Any = raw.get("prompts", raw)
        if not isinstance(prompts_raw, dict):
            raise ValueError("Prompt YAML prompts section must be a mapping")

        prompts: dict[str, str] = {}
        for key, value in prompts_raw.items():
            k = str(key).strip()
            if not k:
                continue
            prompts[k] = str(value).strip()
        return prompts


_DEFAULT_YAML_PATH = Path(__file__).with_name("prompts_compact.yaml")
_DEFAULT_STORE = PromptStore(yaml_path=_DEFAULT_YAML_PATH)


def get_prompt_store() -> PromptStore:
    """Get the shared default prompt store."""
    return _DEFAULT_STORE


def get_prompt(key: str) -> str:
    """Get prompt text by key from shared store."""
    return _DEFAULT_STORE.get(key)


def list_prompts() -> list[str]:
    """List prompt keys from shared store."""
    return _DEFAULT_STORE.keys()


def reload_prompts() -> None:
    """Force reload shared store from disk."""
    _DEFAULT_STORE.reload()

