from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ..comm.send import TextPushResult, push_text_message
from ..core.observe_qa import ObserveAnswerResult, try_answer_with_rag
from ..flow.engine import Engine
from ..msg.types import MessageEvent
from .memory import AgentMemory
from .models import ToolCallSpec


@dataclass(frozen=True, slots=True)
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    silent_safe: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [item.to_dict() for item in self.parameters],
            "side_effects": list(self.side_effects),
            "silent_safe": self.silent_safe,
        }


TOOL_DEFINITIONS: tuple[AgentToolDefinition, ...] = (
    AgentToolDefinition(
        name="observe_message",
        description="Run the silent message through the LangGraph decision kernel: cache, detect, lift, route, store, and optionally push.",
        parameters=[
            ToolParameter("message", "MessageEvent", "Normalized Feishu message event."),
            ToolParameter("context", "object", "Agent run context.", required=False),
        ],
        side_effects=["chat_history_write", "state_write", "optional_feishu_push"],
        silent_safe=True,
    ),
    AgentToolDefinition(
        name="retrieve_knowledge",
        description="Search long-term vector memory for relevant knowledge.",
        parameters=[
            ToolParameter("query", "string", "Search query."),
            ToolParameter("top_k", "integer", "Maximum hits.", required=False),
            ToolParameter("min_score", "number", "Minimum similarity score.", required=False),
        ],
        side_effects=[],
        silent_safe=True,
    ),
    AgentToolDefinition(
        name="answer_from_memory",
        description="Build a lightweight answer from retrieved knowledge when a message is an explicit question.",
        parameters=[
            ToolParameter("query", "string", "User question or inferred query."),
            ToolParameter("min_hits", "integer", "Minimum hits required to answer.", required=False),
        ],
        side_effects=[],
        silent_safe=True,
    ),
    AgentToolDefinition(
        name="push_text",
        description="Push a text message to Feishu. This is used only after planner/engine confidence gates pass.",
        parameters=[
            ToolParameter("chat_id", "string", "Target Feishu chat id."),
            ToolParameter("text", "string", "Text body to send."),
        ],
        side_effects=["feishu_push"],
        silent_safe=False,
    ),
)


@dataclass(slots=True)
class AgentTools:
    engine: Engine
    memory: AgentMemory
    env_file: str = ""

    @property
    def definitions(self) -> list[AgentToolDefinition]:
        return list(TOOL_DEFINITIONS)

    def describe_tools(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.definitions]

    def execute(self, call: ToolCallSpec) -> Any:
        registry: dict[str, Callable[..., Any]] = {
            "observe_message": self.observe_message,
            "retrieve_knowledge": self.retrieve_knowledge,
            "answer_from_memory": self.answer_from_memory,
            "push_text": self.push_text,
        }
        tool = registry.get(call.tool_name)
        if tool is None:
            raise ValueError(f"unknown agent tool: {call.tool_name}")
        return tool(**call.args)

    def observe_message(self, message: MessageEvent, *, context: dict[str, Any] | None = None):
        return self.engine.run(message, context=context or {})

    def retrieve_knowledge(self, query: str, *, top_k: int | None = None, min_score: float | None = None) -> list:
        return self.memory.search_knowledge(query, top_k=top_k, min_score=min_score)

    def answer_from_memory(self, query: str, *, min_hits: int = 1) -> ObserveAnswerResult:
        hits = self.retrieve_knowledge(query)
        return try_answer_with_rag(query, hits, min_hits=min_hits)

    def push_text(self, *, chat_id: str, text: str) -> TextPushResult:
        return push_text_message(chat_id=chat_id, text=text, env_file=self.env_file)


__all__ = ["AgentToolDefinition", "AgentTools", "TOOL_DEFINITIONS", "ToolParameter"]
