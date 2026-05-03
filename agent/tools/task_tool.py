from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime

from insight.core.task import save_task
from insight.core.task import TaskPushConfig
from agent.state_manager import AgentStateManager


class TaskCreateInput(BaseModel):
    """创建任务输入"""
    title: str = Field(description="任务标题")
    description: str = Field(description="任务描述")
    assignee: Optional[str] = Field(None, description="负责人")
    priority: str = Field(default="medium", description="优先级: low/medium/high")
    due_date: Optional[datetime] = Field(None, description="截止日期")
    tags: List[str] = Field(default_factory=list, description="任务标签")
    push_notification: bool = Field(default=False, description="是否推送通知")
    push_chat_id: Optional[str] = Field(None, description="推送的聊天ID")


class TaskUpdateInput(BaseModel):
    """更新任务状态输入"""
    task_id: str = Field(description="任务ID")
    status: str = Field(description="任务状态: pending/in_progress/completed/cancelled")
    comment: Optional[str] = Field(None, description="更新备注")


class TaskGetInput(BaseModel):
    """获取任务输入"""
    task_id: str = Field(description="任务ID")


class TaskDeleteInput(BaseModel):
    """删除任务输入"""
    task_id: str = Field(description="任务ID")


class TaskListInput(BaseModel):
    """列出任务输入"""
    limit: int = Field(default=20, description="返回数量")
    offset: int = Field(default=0, description="偏移量")
    status_filter: Optional[str] = Field(None, description="状态过滤")
    assignee_filter: Optional[str] = Field(None, description="负责人过滤")


class TaskTool:
    """任务管理工具"""

    def __init__(self, state_manager: AgentStateManager, env_file: str = ".env"):
        self.state_manager = state_manager
        self.state_store = state_manager.state_store
        self.identity_map = state_manager.identity_map
        self.env_file = env_file

    def create(self, input_data: TaskCreateInput) -> Dict[str, Any]:
        """创建新的任务"""
        from insight.shared.models import LiftedCard

        card = LiftedCard(
            card_id=f"task_{input_data.title[:20]}_{hash(input_data.description)}",
            candidate_id=f"cand_task_{input_data.title[:20]}",
            title=input_data.title,
            summary=input_data.description,
            problem=input_data.description,
            suggestion=f"优先级: {input_data.priority}\n负责人: {input_data.assignee or '未分配'}\n截止日期: {input_data.due_date or '未设置'}",
            participants=[input_data.assignee] if input_data.assignee else [],
            times=input_data.due_date.isoformat() if input_data.due_date else "",
            locations="",
            evidence=[],
            tags=input_data.tags + [f"priority:{input_data.priority}"],
            confidence=0.9,
            suggested_target="task",
            source_message_ids=[],
        )

        push_config = TaskPushConfig(
            enabled=input_data.push_notification,
            chat_id=input_data.push_chat_id or "",
            env_file=self.env_file
        )

        task_result = save_task(
            store=self.state_store,
            card=card,
            run_id=f"agent_task_{datetime.now().timestamp()}",
            push_config=push_config,
            source_chat_id="agent_call",
            identity_map=self.identity_map
        )

        return {
            "success": True,
            "task_id": task_result.task_id,
            "push_status": task_result.push_attempt.status if task_result.push_attempt else "not_sent",
            "message": "任务创建成功"
        }

    def update_status(self, input_data: TaskUpdateInput) -> Dict[str, Any]:
        """更新任务状态"""
        # TODO: 待实现update_task_status方法
        try:
            task = self.state_store.get(f"task:{input_data.task_id}")
            if task:
                if hasattr(task, 'status'):
                    task.status = input_data.status
                elif isinstance(task, dict):
                    task['status'] = input_data.status
                if input_data.comment:
                    if hasattr(task, 'comment'):
                        task.comment = input_data.comment
                    elif isinstance(task, dict):
                        task['comment'] = input_data.comment
                self.state_store.set(f"task:{input_data.task_id}", task)
                success = True
            else:
                success = False
        except Exception:
            success = False

        return {
            "success": success,
            "message": "任务状态更新成功" if success else "任务状态更新失败"
        }

    def get(self, input_data: TaskGetInput) -> Dict[str, Any]:
        """获取单个任务详情"""
        task = self.state_store.get(f"task:{input_data.task_id}")
        if not task:
            return {
                "success": False,
                "message": "任务不存在"
            }

        return {
            "success": True,
            "task": task.to_dict() if hasattr(task, 'to_dict') else task
        }

    def delete(self, input_data: TaskDeleteInput) -> Dict[str, Any]:
        """删除任务"""
        # TODO: 待实现delete_task方法
        try:
            self.state_store.delete(f"task:{input_data.task_id}")
            success = True
        except Exception:
            success = False

        return {
            "success": success,
            "message": "任务删除成功" if success else "任务删除失败"
        }

    def list(self, input_data: TaskListInput) -> Dict[str, Any]:
        """列出任务列表"""
        # TODO: 待实现list_tasks方法
        task_keys = self.state_store.list_keys("task:")
        tasks = []
        for key in task_keys[input_data.offset : input_data.offset + input_data.limit]:
            task = self.state_store.get(key)
            if task:
                # 应用过滤
                if input_data.status_filter:
                    if hasattr(task, 'status') and task.status != input_data.status_filter:
                        continue
                    if isinstance(task, dict) and task.get('status') != input_data.status_filter:
                        continue
                if input_data.assignee_filter:
                    if hasattr(task, 'assignee') and task.assignee != input_data.assignee_filter:
                        continue
                    if isinstance(task, dict) and task.get('assignee') != input_data.assignee_filter:
                        continue
                tasks.append(task.to_dict() if hasattr(task, 'to_dict') else task)

        return {
            "success": True,
            "count": len(tasks),
            "tasks": tasks
        }
