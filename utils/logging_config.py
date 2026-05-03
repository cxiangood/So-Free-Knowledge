from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


DEFAULT_LOG_FORMAT = "[%(app_name)s] [%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _AppNameFilter(logging.Filter):
    def __init__(self, app_name: str) -> None:
        super().__init__()
        self.app_name = app_name

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "app_name"):
            record.app_name = self.app_name
        return True


def normalize_log_level(level: str | int | None, default: str = "INFO") -> int:
    if isinstance(level, int):
        return level
    value = str(level or default).strip().upper()
    if not value:
        value = default
    numeric_level = logging.getLevelName(value)
    if isinstance(numeric_level, int):
        return numeric_level
    raise ValueError(f"Invalid log level: {level!r}")


def configure_logging(
    *,
    level: str | int | None = None,
    log_file: str | os.PathLike[str] | None = None,
    app_name: str = "SOFREE",
    quiet: bool = False,
    force: bool = False,
) -> logging.Logger:
    """Configure project-wide logging.

    Libraries should call ``logging.getLogger(__name__)`` only; process entrypoints
    call this function once so stdout remains reserved for command output.
    """

    resolved_level = normalize_log_level(level or os.getenv("SOFREE_LOG_LEVEL") or "INFO")
    resolved_log_file = str(log_file or os.getenv("SOFREE_LOG_FILE") or "").strip()
    root_logger = logging.getLogger()

    if force:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()

    root_logger.setLevel(resolved_level)
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    app_filter = _AppNameFilter(str(app_name or "SOFREE").strip() or "SOFREE")

    if not quiet and not _has_handler(root_logger, "_sofree_console"):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(app_filter)
        console_handler._sofree_console = True  # type: ignore[attr-defined]
        root_logger.addHandler(console_handler)

    if resolved_log_file:
        log_path = Path(resolved_log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not _has_file_handler(root_logger, log_path):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(app_filter)
            file_handler._sofree_file_path = str(log_path.resolve())  # type: ignore[attr-defined]
            root_logger.addHandler(file_handler)

    if not root_logger.handlers:
        null_handler = logging.NullHandler()
        null_handler._sofree_null = True  # type: ignore[attr-defined]
        root_logger.addHandler(null_handler)

    for handler in root_logger.handlers:
        handler.setLevel(resolved_level)

    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)


def _has_handler(logger: logging.Logger, marker: str) -> bool:
    return any(bool(getattr(handler, marker, False)) for handler in logger.handlers)


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    resolved = str(path.resolve())
    return any(getattr(handler, "_sofree_file_path", "") == resolved for handler in logger.handlers)
