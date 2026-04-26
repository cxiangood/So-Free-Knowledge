import os
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
