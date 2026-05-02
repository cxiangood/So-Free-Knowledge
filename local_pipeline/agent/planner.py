from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..flow.engine import EngineResult
from ..msg.types import MessageEvent
from .models import AgentDecision, AgentTrigger, ToolCallSpec


@dataclass(slots=True)
class SilentKnowledgePlanner:
    """Explicit next-step policy for the silent knowledge Agent."""

    agent_name: str

    def plan_message_observation(
        self,
        message: MessageEvent,
        *,
        trigger: AgentTrigger,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> list[ToolCallSpec]:
        return [
            ToolCallSpec(
                tool_name="observe_message",
                args={
                    "message": message,
                    "context": {
                        "agent_name": self.agent_name,
                        "agent_mode": "silent_knowledge_push",
                        "trigger": trigger,
                        "source": source,
                        **(context or {}),
                    },
                },
                reason="silent message arrived; inspect context and route without requiring a user command",
            )
        ]

    def decide_after_observation(self, result: EngineResult) -> list[AgentDecision]:
        if result.skipped:
            return [
                AgentDecision(
                    action="wait",
                    tool_name="none",
                    reason="message was already processed; no further action",
                    metadata={"message_id": result.message_id},
                )
            ]
        if result.candidate_count <= 0:
            return [
                AgentDecision(
                    action="wait",
                    tool_name="none",
                    reason="no strong knowledge/task/question signal; keep observing silently",
                    metadata={"message_id": result.message_id},
                )
            ]

        decisions: list[AgentDecision] = []
        route_counts = dict(result.routed_counts)
        if route_counts.get("knowledge", 0) > 0:
            decisions.append(
                AgentDecision(
                    action="store",
                    tool_name="observe_message",
                    reason="engine routed lifted signal to knowledge memory",
                    metadata={"count": route_counts.get("knowledge", 0)},
                )
            )
        if route_counts.get("observe", 0) > 0:
            decisions.append(
                AgentDecision(
                    action="wait",
                    tool_name="observe_message",
                    reason="weak signal entered observe pool; wait for fermentation before pushing",
                    metadata={"count": route_counts.get("observe", 0)},
                )
            )
        if route_counts.get("task", 0) > 0:
            decisions.append(
                AgentDecision(
                    action="store",
                    tool_name="observe_message",
                    reason="engine routed lifted signal to task store",
                    metadata={"count": route_counts.get("task", 0)},
                )
            )
        if result.observe_answered_count > 0:
            decisions.append(
                AgentDecision(
                    action="push",
                    tool_name="push_text",
                    reason="explicit question matched reliable memory; answer was pushed",
                    metadata={"count": result.observe_answered_count},
                )
            )
        if result.task_push_sent > 0:
            decisions.append(
                AgentDecision(
                    action="push",
                    tool_name="observe_message",
                    reason="task confidence passed push gate; task card was sent",
                    metadata={"count": result.task_push_sent},
                )
            )
        if result.task_push_failed > 0:
            decisions.append(
                AgentDecision(
                    action="retry",
                    tool_name="observe_message",
                    reason="push failed and was queued for retry",
                    metadata={"count": result.task_push_failed, "errors": list(result.errors)},
                )
            )
        if not decisions:
            decisions.append(
                AgentDecision(
                    action="wait",
                    tool_name="none",
                    reason="signal was processed but no push/store follow-up was required",
                    metadata={"message_id": result.message_id},
                )
            )
        return decisions

    def plan_direct_question(self, query: str) -> list[ToolCallSpec]:
        return [
            ToolCallSpec(
                tool_name="answer_from_memory",
                args={"query": query},
                reason="direct question path uses memory retrieval before any optional push",
            )
        ]


__all__ = ["SilentKnowledgePlanner"]
