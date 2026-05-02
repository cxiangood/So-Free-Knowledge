from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..flow.engine import Engine, EngineConfig
from ..msg.parse import plain_message_to_event
from ..msg.types import MessageEvent, PlainMessage
from .memory import AgentMemory, AgentMemoryConfig
from .models import AgentBatchReport, AgentRunReport, AgentTrigger
from .planner import SilentKnowledgePlanner
from .tools import AgentTools


@dataclass(slots=True)
class SilentKnowledgeAgentConfig:
    name: str = "sofree-silent-knowledge-agent"
    description: str = "Silently observes Feishu conversations and pushes knowledge only when signals are strong enough."
    engine: EngineConfig = field(default_factory=EngineConfig)

    @classmethod
    def from_engine_config(cls, engine_config: EngineConfig, *, name: str = "") -> "SilentKnowledgeAgentConfig":
        return cls(name=name or "sofree-silent-knowledge-agent", engine=engine_config)


class SilentKnowledgeAgent:
    """Agent facade for silent knowledge push.

    The existing LangGraph engine remains the decision kernel. This runtime
    makes the product boundary explicit: messages arrive without user commands,
    memory is consulted/updated by tools, and pushes happen only through routed
    actions inside the engine.
    """

    def __init__(self, config: SilentKnowledgeAgentConfig | None = None) -> None:
        self.config = config or SilentKnowledgeAgentConfig()
        self.engine = Engine(self.config.engine)
        self.memory = AgentMemory(AgentMemoryConfig.from_engine_config(self.config.engine))
        self.tools = AgentTools(engine=self.engine, memory=self.memory, env_file=self.config.engine.env_file)
        self.planner = SilentKnowledgePlanner(agent_name=self.config.name)

    @property
    def name(self) -> str:
        return self.config.name

    def handle_message(
        self,
        message: MessageEvent,
        *,
        source: str = "feishu_event",
        trigger: AgentTrigger = "silent_message",
        context: dict[str, Any] | None = None,
    ) -> AgentRunReport:
        tool_calls = self.planner.plan_message_observation(
            message,
            trigger=trigger,
            source=source,
            context=context,
        )
        result = self.tools.execute(tool_calls[0])
        decisions = self.planner.decide_after_observation(result)
        return AgentRunReport(
            agent_name=self.name,
            trigger=trigger,
            source=source,
            message_id=message.message_id,
            chat_id=message.chat_id,
            engine_result=result,
            tool_calls=tool_calls,
            decisions=decisions,
        )

    def handle_plain_message(
        self,
        message: PlainMessage,
        *,
        source: str = "offline_archive",
        trigger: AgentTrigger = "manual_replay",
    ) -> AgentRunReport:
        return self.handle_message(plain_message_to_event(message), source=source, trigger=trigger)

    def handle_plain_messages(
        self,
        messages: Iterable[PlainMessage],
        *,
        source: str = "offline_archive",
        trigger: AgentTrigger = "manual_replay",
    ) -> AgentBatchReport:
        reports = [self.handle_plain_message(message, source=source, trigger=trigger) for message in messages]
        return AgentBatchReport(agent_name=self.name, trigger=trigger, source=source, reports=reports)

    def answer_question(self, query: str, *, chat_id: str = "", push: bool = False) -> dict[str, Any]:
        tool_calls = self.planner.plan_direct_question(query)
        answer = self.tools.execute(tool_calls[0])
        pushed = None
        if push and answer.can_answer and chat_id:
            pushed = self.tools.push_text(chat_id=chat_id, text=answer.answer).to_dict()
        return {
            "ok": True,
            "agent_name": self.name,
            "trigger": "direct_question",
            "can_answer": answer.can_answer,
            "answer": answer.answer,
            "reason": answer.reason,
            "hit_count": len(answer.hits or []),
            "tool_calls": [item.to_dict() for item in tool_calls],
            "pushed": pushed,
        }

    def tool_definitions(self) -> list[dict[str, Any]]:
        return self.tools.describe_tools()


__all__ = ["SilentKnowledgeAgent", "SilentKnowledgeAgentConfig"]
