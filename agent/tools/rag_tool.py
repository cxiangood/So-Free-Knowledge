from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field

from insight.core.observe_qa import try_answer_with_rag
from insight.rag.retriever import retrieve
from agent.state_manager import AgentStateManager


class RAGQueryInput(BaseModel):
    """RAG查询输入"""
    query: str = Field(description="用户问题")
    top_k: int = Field(default=5, description="检索文档数量")
    min_score: float = Field(default=0.35, description="最低匹配分数")
    return_sources: bool = Field(default=True, description="是否返回来源信息")


class RAGAnswerInput(BaseModel):
    """RAG回答生成输入"""
    query: str = Field(description="用户问题")
    context_docs: Optional[List[Dict[str, Any]]] = Field(None, description="上下文文档，如果不提供则自动检索")
    top_k: int = Field(default=5, description="检索文档数量")
    min_score: float = Field(default=0.35, description="最低匹配分数")


class RAGRetrieveInput(BaseModel):
    """仅检索文档输入"""
    query: str = Field(description="查询关键词")
    top_k: int = Field(default=10, description="返回结果数量")
    min_score: float = Field(default=0.3, description="最低匹配分数")


class RAGTool:
    """RAG问答工具"""

    def __init__(self, state_manager: AgentStateManager):
        self.state_manager = state_manager
        self.vector_store = state_manager.vector_store

    def query(self, input_data: RAGQueryInput) -> Dict[str, Any]:
        """RAG查询，返回相关文档"""
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

        response = {
            "success": True,
            "count": len(results),
            "results": []
        }

        for hit in results:
            result_item = {
                "score": hit.score,
                "content": hit.content,
            }
            if input_data.return_sources:
                result_item["metadata"] = hit.metadata
            response["results"].append(result_item)

        return response

    def answer(self, input_data: RAGAnswerInput) -> Dict[str, Any]:
        """生成RAG回答"""
        if not self.vector_store:
            return {
                "success": False,
                "message": "RAG功能未启用",
                "can_answer": False,
                "answer": ""
            }

        # 如果没有提供上下文，则自动检索
        context_docs = input_data.context_docs
        if not context_docs:
            retrieve_result = self.query(RAGQueryInput(
                query=input_data.query,
                top_k=input_data.top_k,
                min_score=input_data.min_score,
                return_sources=True
            ))
            if not retrieve_result["success"]:
                return retrieve_result
            context_docs = retrieve_result["results"]

        # 构造hits格式
        class Hit:
            def __init__(self, data):
                self.score = data.get("score", 0.0)
                self.content = data.get("content", "")
                self.metadata = data.get("metadata", {})

        hits = [Hit(doc) for doc in context_docs]
        answer_result = try_answer_with_rag(input_data.query, hits)

        return {
            "success": True,
            "can_answer": answer_result.can_answer,
            "answer": answer_result.answer,
            "confidence": answer_result.confidence,
            "source_count": len(answer_result.hits or []),
            "sources": [
                {
                    "score": hit.score,
                    "title": hit.metadata.get("title", ""),
                    "source": hit.metadata.get("source", "")
                }
                for hit in (answer_result.hits or [])
            ] if input_data.return_sources else []
        }

    def retrieve(self, input_data: RAGRetrieveInput) -> Dict[str, Any]:
        """仅检索相关文档，不生成回答"""
        return self.query(RAGQueryInput(
            query=input_data.query,
            top_k=input_data.top_k,
            min_score=input_data.min_score,
            return_sources=True
        ))
