from __future__ import annotations

from typing import Any, Dict, List, Optional, AsyncGenerator, Generator
from pathlib import Path
import logging
from agent.state_manager import AgentStateManager
from agent.tool_registry import ToolRegistry
from insight.flow.engine import Engine, EngineConfig
from insight.msg.types import MessageEvent

ANTHROPIC_SDK_AVAILABLE = True

class AgentContext:
    def __init__(self, session_id: str = ""):
        self.session_id = session_id

class ToolCall:
    def __init__(self, name: str, parameters: Dict[str, Any], id: str = ""):
        self.name = name
        self.parameters = parameters
        self.id = id

class ToolResult:
    def __init__(self, content: Any, tool_call_id: str = "", is_error: bool = False):
        self.content = content
        self.tool_call_id = tool_call_id
        self.is_error = is_error

class AgentResponse:
    def __init__(
        self,
        content: str = "",
        tool_calls: List[ToolCall] = None,
        tool_results: List[ToolResult] = None,
        metadata: Dict[str, Any] = None
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.tool_results = tool_results or []
        self.metadata = metadata or {}




LOGGER = logging.getLogger(__name__)


class SoFreeKnowledgeAgent():
    """So-Free-Knowledge Harness Agent
    符合Anthropic Agent SDK规范的通用Agent实现
    """

    def __init__(
        self,
        state_dir: str | Path = "outputs/agent_state",
        env_file: str = ".env",
        rag_enabled: bool = True,
        rag_embed_model: str = "BAAI/bge-large-zh",
        chat_history_limit: int = 100,
        task_push_enabled: bool = False,
        task_push_chat_id: str = "",
    ):
        super().__init__()

        # 初始化状态管理
        self.state_manager = AgentStateManager(
            state_dir=state_dir,
            chat_history_limit=chat_history_limit,
            rag_enabled=rag_enabled,
            rag_embed_model=rag_embed_model,
            env_file=env_file
        )

        # 初始化工具注册中心
        self.tool_registry = ToolRegistry(
            state_manager=self.state_manager,
            env_file=env_file
        )

        # 初始化原有Engine，支持完整的消息处理流程
        self.engine = Engine(
            EngineConfig(
                output_dir=str(state_dir),
                state_dir=str(Path(state_dir) / "engine_state"),
                chat_history_path=str(self.state_manager.chat_store.path),
                chat_history_limit=chat_history_limit,
                task_push_enabled=task_push_enabled,
                task_push_chat_id=task_push_chat_id,
                env_file=env_file,
                rag_enabled=rag_enabled,
                rag_embed_model=rag_embed_model,
            )
        )
        self.env_file = env_file

    @property
    def name(self) -> str:
        return "so-free-knowledge-agent"

    @property
    def description(self) -> str:
        return "飞书知识助理Agent，支持消息分析、知识管理、任务管理、RAG问答等能力"

    def get_tools(self) -> List[Dict[str, Any]]:
        """获取Agent支持的所有工具定义，符合OpenAI Function Calling格式"""
        tools = []
        for tool_def in self.tool_registry.list_tools():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": tool_def.input_schema.model_json_schema()
                }
            })
        return tools

    def invoke(
        self,
        input_data: Dict[str, Any] | str | MessageEvent,
        context: Optional[AgentContext] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        标准Agent调用接口
        支持三种输入格式：
        1. MessageEvent: 直接处理飞书消息事件，走完整pipeline
        2. ToolCall格式: {"tool_call": {"name": "xxx", "parameters": {...}}}
        3. 自然语言字符串: 自动路由到合适的工具
        """
        context = context or AgentContext()
        config = config or {}

        try:
            # 情况1: 输入是MessageEvent，走完整处理流程
            if isinstance(input_data, MessageEvent):
                return self._process_message_event(input_data, context)

            # 情况2: 输入是工具调用请求
            if isinstance(input_data, dict) and "tool_call" in input_data:
                return self._handle_tool_call(input_data["tool_call"], context)

            # 情况3: 输入是直接的工具调用参数（简化格式）
            if isinstance(input_data, dict) and "name" in input_data and "parameters" in input_data:
                return self._handle_tool_call(input_data, context)

            # 情况4: 输入是自然语言，简单路由处理
            if isinstance(input_data, str):
                return self._handle_natural_language(input_data, context)

            return AgentResponse(
                content="不支持的输入格式",
                metadata={"success": False, "error": "Unsupported input format"}
            )

        except Exception as e:
            LOGGER.exception(f"Agent invocation failed: {e}")
            return AgentResponse(
                content=f"调用失败: {str(e)}",
                metadata={"success": False, "error": str(e)}
            )

    def stream(
        self,
        input_data: Dict[str, Any] | str | MessageEvent,
        context: Optional[AgentContext] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Generator[AgentResponse, None, None]:
        """流式调用接口，目前返回完整结果"""
        yield self.invoke(input_data, context, config)

    async def ainvoke(
        self,
        input_data: Dict[str, Any] | str | MessageEvent,
        context: Optional[AgentContext] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """异步调用接口"""
        return self.invoke(input_data, context, config)

    async def astream(
        self,
        input_data: Dict[str, Any] | str | MessageEvent,
        context: Optional[AgentContext] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[AgentResponse, None]:
        """异步流式调用接口"""
        yield self.invoke(input_data, context, config)

    def _process_message_event(self, message: MessageEvent, context: AgentContext) -> AgentResponse:
        """处理飞书消息事件，走完整pipeline"""
        engine_result = self.engine.run(message)
        return AgentResponse(
            content=f"消息处理完成，检测到{engine_result.candidate_count}个有效信号",
            metadata={
                "success": True,
                "engine_result": engine_result.to_dict(),
                "session_id": context.session_id
            }
        )

    def _handle_tool_call(self, tool_call: Dict[str, Any] | ToolCall, context: AgentContext) -> AgentResponse:
        """处理工具调用"""
        if isinstance(tool_call, dict):
            tool_call = ToolCall(
                name=tool_call["name"],
                parameters=tool_call.get("parameters", {}),
                id=tool_call.get("id", "")
            )

        result = self.tool_registry.invoke_tool(tool_call.name, tool_call.parameters)

        if result.get("success", False):
            return AgentResponse(
                tool_results=[
                    ToolResult(
                        content=result,
                        tool_call_id=tool_call.id,
                        is_error=False
                    )
                ],
                metadata={"success": True, "session_id": context.session_id}
            )
        else:
            return AgentResponse(
                tool_results=[
                    ToolResult(
                        content=result.get("error", "Unknown error"),
                        tool_call_id=tool_call.id,
                        is_error=True
                    )
                ],
                metadata={"success": False, "error": result.get("error", "Unknown error")}
            )

    def _handle_natural_language(self, query: str, context: AgentContext) -> AgentResponse:
        """处理自然语言请求，简单路由"""
        # 优先尝试用RAG回答
        rag_result = self.tool_registry.invoke_tool("rag_answer", {"query": query})
        if rag_result.get("success", False) and rag_result.get("can_answer", False):
            return AgentResponse(
                content=rag_result["answer"],
                metadata={
                    "success": True,
                    "source_count": rag_result.get("source_count", 0),
                    "sources": rag_result.get("sources", []),
                    "session_id": context.session_id
                }
            )

        # 如果RAG不能回答，尝试分析是否包含任务或知识
        analyze_result = self.tool_registry.invoke_tool(
            "message_analyze",
            {
                "message_content": query,
                "chat_id": context.session_id or "agent_chat",
                "include_context": False
            }
        )

        if analyze_result.get("success", False):
            if analyze_result.get("has_signal", False):
                cards = analyze_result.get("cards", [])
                decisions = analyze_result.get("routing_decisions", [])
                return AgentResponse(
                    content=f"分析完成，检测到{len(cards)}个语义卡片，路由到{len(decisions)}个处理池",
                    metadata={
                        "success": True,
                        "analysis_result": analyze_result,
                        "session_id": context.session_id
                    }
                )

        return AgentResponse(
            content="我是So-Free-Knowledge知识助理，我可以帮你管理知识、处理任务、分析消息。请问你需要什么帮助？",
            metadata={"success": True, "session_id": context.session_id}
        )
