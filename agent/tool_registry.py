from __future__ import annotations

from typing import Any, Dict, List, Type, Callable
from pydantic import BaseModel

from agent.state_manager import AgentStateManager
from agent.tools.knowledge_tool import KnowledgeTool
from agent.tools.task_tool import TaskTool
from agent.tools.message_tool import MessageTool
from agent.tools.rag_tool import RAGTool


class ToolDefinition(BaseModel):
    """工具定义"""
    name: str
    description: str
    input_schema: Type[BaseModel]
    implementation: Callable
    category: str = "general"


class ToolRegistry:
    """工具注册中心"""

    def __init__(self, state_manager: AgentStateManager, env_file: str = ".env"):
        self.state_manager = state_manager
        self.env_file = env_file

        # 初始化工具实例
        self.knowledge_tool = KnowledgeTool(state_manager)
        self.task_tool = TaskTool(state_manager, env_file)
        self.message_tool = MessageTool(state_manager)
        self.rag_tool = RAGTool(state_manager)

        # 注册所有工具
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        """注册所有工具"""
        from agent.tools.knowledge_tool import KnowledgeCreateInput, KnowledgeQueryInput, KnowledgeGetInput, KnowledgeDeleteInput, KnowledgeListInput
        from agent.tools.task_tool import TaskCreateInput, TaskUpdateInput, TaskGetInput, TaskDeleteInput, TaskListInput
        from agent.tools.message_tool import MessageAnalyzeInput, SignalDetectInput, SemanticExtractInput
        from agent.tools.rag_tool import RAGQueryInput, RAGAnswerInput, RAGRetrieveInput

        # 知识管理工具
        self._register_tool(
            name="knowledge_create",
            description="创建新的知识条目，存储到知识库中",
            input_schema=KnowledgeCreateInput,
            implementation=self.knowledge_tool.create,
            category="knowledge"
        )

        self._register_tool(
            name="knowledge_query",
            description="查询知识库，返回相关的知识条目",
            input_schema=KnowledgeQueryInput,
            implementation=self.knowledge_tool.query,
            category="knowledge"
        )

        self._register_tool(
            name="knowledge_get",
            description="根据ID获取单个知识条目的详细信息",
            input_schema=KnowledgeGetInput,
            implementation=self.knowledge_tool.get,
            category="knowledge"
        )

        self._register_tool(
            name="knowledge_delete",
            description="删除指定的知识条目",
            input_schema=KnowledgeDeleteInput,
            implementation=self.knowledge_tool.delete,
            category="knowledge"
        )

        self._register_tool(
            name="knowledge_list",
            description="列出所有知识条目，支持分页和过滤",
            input_schema=KnowledgeListInput,
            implementation=self.knowledge_tool.list,
            category="knowledge"
        )

        # 任务管理工具
        self._register_tool(
            name="task_create",
            description="创建新的任务，支持自动推送通知",
            input_schema=TaskCreateInput,
            implementation=self.task_tool.create,
            category="task"
        )

        self._register_tool(
            name="task_update_status",
            description="更新任务的状态",
            input_schema=TaskUpdateInput,
            implementation=self.task_tool.update_status,
            category="task"
        )

        self._register_tool(
            name="task_get",
            description="根据ID获取任务详情",
            input_schema=TaskGetInput,
            implementation=self.task_tool.get,
            category="task"
        )

        self._register_tool(
            name="task_delete",
            description="删除指定任务",
            input_schema=TaskDeleteInput,
            implementation=self.task_tool.delete,
            category="task"
        )

        self._register_tool(
            name="task_list",
            description="列出任务列表，支持过滤和分页",
            input_schema=TaskListInput,
            implementation=self.task_tool.list,
            category="task"
        )

        # 消息处理工具
        self._register_tool(
            name="message_analyze",
            description="完整分析消息，包括信号检测、语义提取和路由决策",
            input_schema=MessageAnalyzeInput,
            implementation=self.message_tool.analyze,
            category="message"
        )

        self._register_tool(
            name="message_detect_signal",
            description="检测消息中是否包含需要处理的信号",
            input_schema=SignalDetectInput,
            implementation=self.message_tool.detect_signal,
            category="message"
        )

        self._register_tool(
            name="message_extract_semantics",
            description="从消息中提取结构化语义信息",
            input_schema=SemanticExtractInput,
            implementation=self.message_tool.extract_semantics,
            category="message"
        )

        self._register_tool(
            name="message_route",
            description="对消息提取的语义卡片进行路由决策",
            input_schema=SemanticExtractInput,
            implementation=self.message_tool.route_cards,
            category="message"
        )

        # RAG工具
        self._register_tool(
            name="rag_query",
            description="查询知识库返回相关文档",
            input_schema=RAGQueryInput,
            implementation=self.rag_tool.query,
            category="rag"
        )

        self._register_tool(
            name="rag_answer",
            description="基于知识库内容回答用户问题",
            input_schema=RAGAnswerInput,
            implementation=self.rag_tool.answer,
            category="rag"
        )

        self._register_tool(
            name="rag_retrieve",
            description="仅检索相关文档，不生成回答",
            input_schema=RAGRetrieveInput,
            implementation=self.rag_tool.retrieve,
            category="rag"
        )

    def _register_tool(
        self,
        name: str,
        description: str,
        input_schema: Type[BaseModel],
        implementation: Callable,
        category: str = "general"
    ) -> None:
        """注册单个工具"""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            implementation=implementation,
            category=category
        )

    def get_tool(self, name: str) -> ToolDefinition | None:
        """获取工具定义"""
        return self._tools.get(name)

    def list_tools(self, category: str | None = None) -> List[ToolDefinition]:
        """列出所有工具，可按类别过滤"""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_tool_schema(self, name: str) -> Dict[str, Any] | None:
        """获取工具的JSON Schema定义"""
        tool = self.get_tool(name)
        if not tool:
            return None
        return tool.input_schema.model_json_schema()

    def invoke_tool(self, name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        tool = self.get_tool(name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool {name} not found"
            }

        try:
            # 验证输入数据
            validated_input = tool.input_schema(**input_data)
            # 执行工具
            result = tool.implementation(validated_input)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }
