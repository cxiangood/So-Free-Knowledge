from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


@dataclass(slots=True)
class PlanRecord:
    plan_id: str
    title: str
    goal: str
    status: str = "draft"
    owner: str = ""
    doc_url: str = ""
    bitable_url: str = ""
    task_urls: list[str] = field(default_factory=list)
    last_openclaw_message_id: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PlanRecord":
        return cls(
            plan_id=str(data["plan_id"]),
            title=str(data["title"]),
            goal=str(data["goal"]),
            status=str(data.get("status", "draft")),
            owner=str(data.get("owner", "")),
            doc_url=str(data.get("doc_url", "")),
            bitable_url=str(data.get("bitable_url", "")),
            task_urls=[str(item) for item in data.get("task_urls", [])],
            last_openclaw_message_id=str(data.get("last_openclaw_message_id", "")),
            created_at=str(data.get("created_at", now_iso())),
            updated_at=str(data.get("updated_at", now_iso())),
        )
