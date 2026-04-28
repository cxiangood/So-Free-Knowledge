from __future__ import annotations

from datetime import datetime, timezone


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["now_utc_iso"]
