from __future__ import annotations

import logging
import os
import sys
from threading import Lock
from collections import OrderedDict


LOGGER = logging.getLogger(__name__)
_TRACE_LOCK = Lock()
_TRACE_STATE: dict[str, dict[str, str | list[str]]] = {}

# 动态任务状态追踪
_TRACKING_TASKS: OrderedDict[str, dict] = OrderedDict()
_MAX_DISPLAY_TASKS = 10  # 最多同时显示10个任务
_STEP_NAMES = {
    "message_cache": "消息缓存",
    "deduplicate": "去重判断",
    "context_extract": "上下文提取",
    "signal_detect": "信号检测",
    "semantic_lift": "语义升维",
    "route": "路由判断",
    "knowledge_store": "知识存储",
    "task_store": "任务存储",
    "observe_store": "观察存储",
    "observe_logic1_check": "观察逻辑1",
    "observe_logic2_check": "观察逻辑2",
    "observe_logic3_check": "观察逻辑3",
    "rag_retrieve_task": "任务RAG检索",
    "rag_retrieve_observe": "观察RAG检索",
    "observe_reply": "观察自动回复",
    "task_push": "任务推送",
    "observe_pop_route": "观察弹出路由",
    "done": "处理完成",
}


def _clear_screen() -> None:
    """清屏，兼容Windows和Unix系统"""
    os.system('cls' if sys.platform == 'win32' else 'clear')


def _format_step_name(node: str) -> str:
    if node.endswith(")") and "(" in node:
        name, suffix = node.rsplit("(", 1)
        return f"{_STEP_NAMES.get(name, name)}({suffix}"
    return _STEP_NAMES.get(node, node)


def _format_step_path(nodes: list[str]) -> str:
    return " → ".join(_format_step_name(node) for node in nodes)


def _render_progress() -> None:
    """渲染当前所有任务的进度"""
    with _TRACE_LOCK:
        if not _TRACKING_TASKS:
            return

        rows = []
        for task in reversed(_TRACKING_TASKS.values()):
            content = task["content"]
            current_step = task["current_step"]
            status = "完成" if task["completed"] else "执行中"
            rows.append(f"[{status}] {content}")
            rows.append(f"{current_step}")
            rows.append("")

        _clear_screen()
        print("实时任务进度")
        for row in rows:
            print(row)
        print(flush=True)


def trace_start(*, message_id: str, chat_id: str, content: str) -> None:
    with _TRACE_LOCK:
        full_content = (content or "").replace("\r", " ").replace("\n", " ").strip()
        short_content = full_content[:27] + "..." if len(full_content) > 30 else full_content
        _TRACE_STATE[message_id] = {
            "chat_id": chat_id,
            "content": short_content,
            "nodes": [],
        }

        # 加入任务追踪
        _TRACKING_TASKS[message_id] = {
            "content": short_content,
            "current_step": "消息缓存",
            "completed": False
        }

        # 限制最多显示任务数
        if len(_TRACKING_TASKS) > _MAX_DISPLAY_TASKS:
            _TRACKING_TASKS.popitem(last=False)

        LOGGER.info(f"[{short_content}] trace started")

    # 刷新进度
    _render_progress()


def trace_node(*, message_id: str, node_name: str) -> None:
    with _TRACE_LOCK:
        state = _TRACE_STATE.get(message_id)
        if not state:
            return
        nodes = state.get("nodes")
        if isinstance(nodes, list):
            nodes.append(node_name)
            path = "→".join(nodes)
            content = str(state.get("content", ""))
            LOGGER.info(f"[{content}] 执行中: {path}")

            current_step = _format_step_path(nodes)

            # 更新任务追踪中的当前步骤完整路径
            if message_id in _TRACKING_TASKS:
                _TRACKING_TASKS[message_id]["current_step"] = current_step

    # 刷新进度
    _render_progress()


def trace_finish(*, message_id: str, suffix_status: str = "ok") -> None:
    with _TRACE_LOCK:
        state = _TRACE_STATE.pop(message_id, None)
        if not state:
            return
        content = str(state.get("content", ""))
        nodes = state.get("nodes", [])
        if not isinstance(nodes, list):
            nodes = []

        # 去除重复的连续节点（解决context_extract出现两次的问题）
        unique_nodes = []
        prev_node = None
        for node in nodes:
            if node != prev_node:
                unique_nodes.append(node)
                prev_node = node

        if suffix_status and suffix_status != "ok" and unique_nodes:
            unique_nodes[-1] = f"{unique_nodes[-1]}({suffix_status})"

        path = "→".join(unique_nodes) if unique_nodes else "无节点"
        line = f"[{content}] 完成: {path}"
        LOGGER.info(line)
        display_path = _format_step_path(unique_nodes) if unique_nodes else "无节点"

        # 更新任务状态为已完成
        if message_id in _TRACKING_TASKS:
            _TRACKING_TASKS[message_id]["completed"] = True
            _TRACKING_TASKS[message_id]["current_step"] = display_path

    # 刷新进度显示最终状态
    _render_progress()


__all__ = ["trace_start", "trace_node", "trace_finish"]
