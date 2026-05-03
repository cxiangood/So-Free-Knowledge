from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from insight.msg.cache import ChatMessageStore
from insight.store.state import LocalStateStore
from insight.rag.index import VectorKnowledgeStore
from insight.comm.identity_map import UserIdentityMap, resolve_identity_map_config


class AgentStateManager:
    """Agent状态管理器，适配原有系统的状态存储"""

    def __init__(
        self,
        state_dir: str | Path = "outputs/agent_state",
        chat_history_limit: int = 100,
        rag_enabled: bool = True,
        rag_embed_model: str = "BAAI/bge-large-zh",
        env_file: str = ".env",
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # 初始化原有状态存储
        self.state_store = LocalStateStore(self.state_dir)
        self.chat_store = ChatMessageStore(
            path=self.state_dir / "chat_message_store.json",
            max_messages_per_chat=chat_history_limit
        )

        # 向量知识库
        self.vector_store: Optional[VectorKnowledgeStore] = None
        if rag_enabled:
            try:
                self.vector_store = VectorKnowledgeStore(
                    root_path=self.state_dir / "vector_kb",
                    embed_model=rag_embed_model
                )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Vector store initialization failed: {e}, RAG capabilities disabled")

        # 用户身份映射
        self.identity_map = UserIdentityMap(
            resolve_identity_map_config(
                state_dir=str(self.state_dir),
                env_file=env_file
            )
        )

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        return self.state_store.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """设置状态"""
        self.state_store.set(key, value)

    def delete_state(self, key: str) -> None:
        """删除状态"""
        self.state_store.delete(key)

    def list_state_keys(self, prefix: str = "") -> list[str]:
        """列出所有状态键"""
        return self.state_store.list_keys(prefix)

    def add_chat_message(self, message) -> None:
        """添加聊天消息到历史"""
        self.chat_store.append(message)

    def get_chat_history(self, chat_id: str, limit: Optional[int] = None) -> list:
        """获取聊天历史"""
        history = self.chat_store.get_chat_messages(chat_id)
        if limit is not None:
            history = history[-limit:]
        return history

    def clear_chat_history(self, chat_id: str) -> None:
        """清空聊天历史"""
        self.chat_store.clear_chat(chat_id)

    def reset(self) -> None:
        """重置所有状态"""
        self.state_store.clear()
        self.chat_store.clear_all()
        if self.vector_store:
            self.vector_store.clear()
