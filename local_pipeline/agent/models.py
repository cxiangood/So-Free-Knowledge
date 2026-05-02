from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..flow.engine import EngineResult
from ..shared.utils import now_utc_iso


AgentTrigger = Literal["silent_message", "manual_replay", "scheduled", "direct_question"]
AgentAction = Literal["observe", "store", "push", "wait", "retry", "answer"]


@dataclass(slots=True)
class ToolCallSpec:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arg_keys": sorted(self.args.keys()),
            "reason": self.reason,
        }


@dataclass(slots=True)
class AgentDecision:
    action: AgentAction
    tool_name: str
    reason: str
    should_execute: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRunReport:
    agent_name: str
    trigger: AgentTrigger
    source: str
    message_id: str
    chat_id: str
    engine_result: EngineResult
    tool_calls: list[ToolCallSpec] = field(default_factory=list)
    decisions: list[AgentDecision] = field(default_factory=list)
    created_at: str = field(default_factory=now_utc_iso)

    @property
    def pushed_count(self) -> int:
        return int(self.engine_result.task_push_sent) + int(self.engine_result.observe_answered_count)

    @property
    def stored_count(self) -> int:
        return sum(int(value) for value in self.engine_result.routed_counts.values())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["engine_result"] = self.engine_result.to_dict()
        payload["tool_calls"] = [item.to_dict() for item in self.tool_calls]
        payload["decisions"] = [item.to_dict() for item in self.decisions]
        payload["pushed_count"] = self.pushed_count
        payload["stored_count"] = self.stored_count
        return payload


@dataclass(slots=True)
class AgentBatchReport:
    agent_name: str
    trigger: AgentTrigger
    source: str
    reports: list[AgentRunReport] = field(default_factory=list)
    created_at: str = field(default_factory=now_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        route_counts: dict[str, int] = {}
        warnings: list[str] = []
        errors: list[str] = []
        totals = {
            "message_count": 0,
            "candidate_count": 0,
            "task_push_attempted": 0,
            "task_push_sent": 0,
            "task_push_failed": 0,
            "rag_retrieval_count": 0,
            "observe_question_count": 0,
            "observe_answered_count": 0,
            "observe_fallback_count": 0,
            "observe_pop_count": 0,
            "observe_reroute_task_count": 0,
            "observe_reroute_knowledge_count": 0,
            "denoise_filtered_count": 0,
        }
        for report in self.reports:
            result = report.engine_result
            totals["message_count"] += 1
            totals["candidate_count"] += int(result.candidate_count)
            totals["task_push_attempted"] += int(result.task_push_attempted)
            totals["task_push_sent"] += int(result.task_push_sent)
            totals["task_push_failed"] += int(result.task_push_failed)
            totals["rag_retrieval_count"] += int(result.rag_retrieval_count)
            totals["observe_question_count"] += int(result.observe_question_count)
            totals["observe_answered_count"] += int(result.observe_answered_count)
            totals["observe_fallback_count"] += int(result.observe_fallback_count)
            totals["observe_pop_count"] += int(result.observe_pop_count)
            totals["observe_reroute_task_count"] += int(result.observe_reroute_task_count)
            totals["observe_reroute_knowledge_count"] += int(result.observe_reroute_knowledge_count)
            totals["denoise_filtered_count"] += int(result.denoise_filtered_count)
            for key, value in result.routed_counts.items():
                route_counts[key] = route_counts.get(key, 0) + int(value)
            warnings.extend(result.warnings)
            errors.extend(result.errors)
        return {
            "ok": True,
            "agent_name": self.agent_name,
            "trigger": self.trigger,
            "source": self.source,
            "route_counts": route_counts,
            **totals,
            "warnings": warnings,
            "errors": errors,
            "created_at": self.created_at,
        }


__all__ = [
    "AgentAction",
    "AgentBatchReport",
    "AgentDecision",
    "AgentRunReport",
    "AgentTrigger",
    "ToolCallSpec",
]
