import os
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv


class EnvManager:
    """环境变量管理器，优先从.env文件读取，失败则从系统环境变量读取"""

    _instance = None
    _loaded = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, dotenv_path: str = ".env"):
        if not self._loaded:
            self.dotenv_path = dotenv_path
            self._load_dotenv()
            self._loaded = True

    def _load_dotenv(self) -> None:
        """加载.env文件"""
        if os.path.exists(self.dotenv_path):
            load_dotenv(self.dotenv_path, override=True)

    def getenv(self, name: str, default: Any = None) -> Optional[str]:
        """
        获取环境变量

        Args:
            name: 环境变量名称
            default: 默认值

        Returns:
            环境变量值，如果不存在则返回default
        """
        return os.getenv(name, default)


# 全局实例
env_manager = EnvManager()


def getenv(name: str, default: Any = None) -> Optional[str]:
    """
    环境变量获取

    Args:
        name: 环境变量名称
        default: 默认值

    Returns:
        环境变量值，如果不存在则返回default
    """
    return env_manager.getenv(name, default)


def load_env_file(path: str | Path | None, override: bool = False) -> None:
    """Load key-value pairs from an env file into process environment."""
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
