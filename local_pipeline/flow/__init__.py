from .engine import Engine, EngineConfig, EngineResult

__all__ = [
    "Engine",
    "EngineConfig",
    "EngineResult",
    "OnlineConfig",
    "start",
    "OfflineConfig",
    "run",
    "DEFAULT_MESSAGES_FILE",
]


def __getattr__(name: str):
    if name in {"DEFAULT_MESSAGES_FILE", "OfflineConfig", "run"}:
        from . import offline

        return getattr(offline, name)
    if name in {"OnlineConfig", "start"}:
        from . import online

        return getattr(online, name)
    raise AttributeError(name)

