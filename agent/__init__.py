"""
So-Free-Knowledge Harness Agent
符合Anthropic Agent SDK规范的通用Agent实现
"""

from .interface import SoFreeKnowledgeAgent
from .state_manager import AgentStateManager
from .tool_registry import ToolRegistry

__version__ = "1.0.0"
__author__ = "So-Free-Knowledge Team"

__all__ = [
    "SoFreeKnowledgeAgent",
    "AgentStateManager",
    "ToolRegistry",
]
