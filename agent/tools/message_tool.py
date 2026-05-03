from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field

from insight.core.detect import detect_candidates
from insight.core.lift import lift_candidates
from insight.core.route import route_cards
from insight.msg.types import MessageEvent
from agent.state_manager import AgentStateManager


class MessageAnalyzeInput(BaseModel):
    """消息分析输入"""
    message_content: str = Field(description="消息内容")
    chat_id: str = Field(description="聊天ID")
    sender_id: Optional[str] = Field(None, description="发送者ID")
    sender_name: Optional[str] = Field(None, description="发送者名称")
    include_context: bool = Field(default=True, description="是否包含上下文历史")
    context_window: int = Field(default=10, description="上下文窗口大小")


class SignalDetectInput(BaseModel):
    """信号检测输入"""
    messages: List[str] = Field(description="消息列表")
    detect_threshold: float = Field(default=45.0, description="检测阈值")


class SemanticExtractInput(BaseModel):
    """语义提取输入"""
    messages: List[str] = Field(description="消息列表")


class MessageTool:
    """消息处理工具"""

    def __init__(self, state_manager: AgentStateManager):
        self.state_manager = state_manager
        self.chat_store = state_manager.chat_store

    def analyze(self, input_data: MessageAnalyzeInput) -> Dict[str, Any]:
        """完整分析单条消息"""
        # 创建消息事件
        message = MessageEvent(
            message_id=f"msg_{hash(input_data.message_content)}_{input_data.chat_id}",
            chat_id=input_data.chat_id,
            sender_id=input_data.sender_id or "unknown",
            sender_name=input_data.sender_name or "unknown",
            content_text=input_data.message_content,
            created_at=None,
            message_type="text",
            mentions=[],
        )

        # 保存到聊天历史
        self.chat_store.append(message)

        # 获取上下文
        if input_data.include_context:
            context_rows = self.chat_store.get_chat_messages(input_data.chat_id)[-input_data.context_window:]
            from insight.msg.parse import event_row_to_message_event
            message_events = [item for item in (event_row_to_message_event(row) for row in context_rows) if item is not None]
            simple_messages = [item.get_simple_message() for item in message_events]
        else:
            simple_messages = [message.get_simple_message()]

        # 信号检测
        detection = detect_candidates(simple_messages)
        detect_score = float(detection.value_score)

        result = {
            "success": True,
            "detect_score": detect_score,
            "has_signal": detect_score >= 45.0,
            "detection": detection.to_dict(),
            "cards": [],
            "routing_decisions": []
        }

        # 如果有信号，进行语义提取和路由
        if detect_score >= 45.0:
            lift_result = lift_candidates(simple_messages)
            result["cards"] = [c.to_dict() for c in lift_result.cards]

            if lift_result.cards:
                decisions = route_cards(lift_result.cards)
                result["routing_decisions"] = [d.to_dict() for d in decisions]

        return result

    def detect_signal(self, input_data: SignalDetectInput) -> Dict[str, Any]:
        """仅检测消息中的信号"""
        detection = detect_candidates(input_data.messages)
        detect_score = float(detection.value_score)

        return {
            "success": True,
            "detect_score": detect_score,
            "has_signal": detect_score >= input_data.detect_threshold,
            "detection": detection.to_dict()
        }

    def extract_semantics(self, input_data: SemanticExtractInput) -> Dict[str, Any]:
        """提取消息语义信息"""
        lift_result = lift_candidates(input_data.messages)

        return {
            "success": True,
            "card_count": len(lift_result.cards),
            "cards": [c.to_dict() for c in lift_result.cards],
            "warnings": lift_result.warnings
        }

    def route_cards(self, input_data: SemanticExtractInput) -> Dict[str, Any]:
        """对提取的卡片进行路由决策"""
        lift_result = lift_candidates(input_data.messages)

        if not lift_result.cards:
            return {
                "success": True,
                "decision_count": 0,
                "decisions": []
            }

        decisions = route_cards(lift_result.cards)

        return {
            "success": True,
            "decision_count": len(decisions),
            "decisions": [d.to_dict() for d in decisions]
        }
