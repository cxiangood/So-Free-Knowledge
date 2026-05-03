from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field

from insight.core.kb import save_knowledge
from insight.rag.retriever import retrieve
from agent.state_manager import AgentStateManager


class KnowledgeCreateInput(BaseModel):
    """创建知识条目输入"""
    title: str = Field(description="知识标题")
    summary: str = Field(description="知识摘要")
    content: str = Field(description="知识详细内容")
    tags: List[str] = Field(default_factory=list, description="知识标签")
    source: Optional[str] = Field(None, description="知识来源")


class KnowledgeQueryInput(BaseModel):
    """查询知识输入"""
    query: str = Field(description="查询关键词")
    top_k: int = Field(default=5, description="返回结果数量")
    min_score: float = Field(default=0.35, description="最低匹配分数")


class KnowledgeGetInput(BaseModel):
    """获取单个知识输入"""
    knowledge_id: str = Field(description="知识ID")


class KnowledgeDeleteInput(BaseModel):
    """删除知识输入"""
    knowledge_id: str = Field(description="知识ID")


class KnowledgeListInput(BaseModel):
    """列出知识输入"""
    limit: int = Field(default=20, description="返回数量")
    offset: int = Field(default=0, description="偏移量")
    tag_filter: Optional[str] = Field(None, description="标签过滤")


class KnowledgeTool:
    """知识管理工具"""

    def __init__(self, state_manager: AgentStateManager):
        self.state_manager = state_manager
        self.state_store = state_manager.state_store
        self.vector_store = state_manager.vector_store

    def create(self, input_data: KnowledgeCreateInput) -> Dict[str, Any]:
        """创建新的知识条目"""
        from insight.shared.models import LiftedCard

        card = LiftedCard(
            card_id=f"kb_{input_data.title[:20]}_{hash(input_data.content)}",
            candidate_id=f"cand_{input_data.title[:20]}",
            title=input_data.title,
            summary=input_data.summary,
            problem="",
            suggestion=input_data.content,
            participants=[],
            times="",
            locations="",
            evidence=[input_data.source] if input_data.source else [],
            tags=input_data.tags,
            confidence=0.9,
            suggested_target="knowledge",
            source_message_ids=[],
        )

        knowledge_id = save_knowledge(
            store=self.state_store,
            card=card,
            vector_store=self.vector_store
        )

        return {
            "success": True,
            "knowledge_id": knowledge_id,
            "message": "知识创建成功"
        }

    def query(self, input_data: KnowledgeQueryInput) -> Dict[str, Any]:
        """查询知识库"""
        if not self.vector_store:
            return {
                "success": False,
                "message": "RAG功能未启用",
                "results": []
            }

        results = retrieve(
            vector_store=self.vector_store,
            query=input_data.query,
            top_k=input_data.top_k,
            min_score=input_data.min_score
        )

        return {
            "success": True,
            "count": len(results),
            "results": [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "title": hit.metadata.get("title", ""),
                    "summary": hit.metadata.get("summary", ""),
                    "content": hit.content,
                    "tags": hit.metadata.get("tags", []),
                    "source": hit.metadata.get("source", "")
                }
                for hit in results
            ]
        }

    def get(self, input_data: KnowledgeGetInput) -> Dict[str, Any]:
        """获取单个知识详情"""
        # TODO: 待实现get_knowledge方法
        knowledge = self.state_store.get(f"knowledge:{input_data.knowledge_id}")
        if not knowledge:
            return {
                "success": False,
                "message": "知识不存在"
            }

        return {
            "success": True,
            "knowledge": knowledge.to_dict() if hasattr(knowledge, 'to_dict') else knowledge
        }

    def delete(self, input_data: KnowledgeDeleteInput) -> Dict[str, Any]:
        """删除知识条目"""
        # TODO: 待实现delete_knowledge方法
        try:
            self.state_store.delete(f"knowledge:{input_data.knowledge_id}")
            if self.vector_store:
                self.vector_store.delete([input_data.knowledge_id])
            success = True
        except Exception:
            success = False

        return {
            "success": success,
            "message": "知识删除成功" if success else "知识删除失败"
        }

    def list(self, input_data: KnowledgeListInput) -> Dict[str, Any]:
        """列出所有知识"""
        # TODO: 待实现list_knowledge方法
        knowledge_keys = self.state_store.list_keys("knowledge:")
        knowledge_list = []
        for key in knowledge_keys[input_data.offset : input_data.offset + input_data.limit]:
            knowledge = self.state_store.get(key)
            if knowledge:
                knowledge_list.append(knowledge.to_dict() if hasattr(knowledge, 'to_dict') else knowledge)

        return {
            "success": True,
            "count": len(knowledge_list),
            "knowledge_list": knowledge_list
        }
