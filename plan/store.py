from __future__ import annotations

import json
from pathlib import Path

from plan.models import PlanRecord, now_iso


class PlanStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[1] / "plans"
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, plan_id: str) -> Path:
        return self.root / f"{plan_id}.json"

    def save(self, record: PlanRecord) -> None:
        record.updated_at = now_iso()
        self.path_for(record.plan_id).write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, plan_id: str) -> PlanRecord:
        path = self.path_for(plan_id)
        if not path.exists():
            raise SystemExit(f"Plan not found: {plan_id}")
        return PlanRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

