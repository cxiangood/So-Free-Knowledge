from __future__ import annotations

import logging
from threading import Lock


LOGGER = logging.getLogger(__name__)
_TRACE_LOCK = Lock()
_TRACE_STATE: dict[str, dict[str, str | list[str]]] = {}


def trace_start(*, message_id: str, chat_id: str, content: str) -> None:
    with _TRACE_LOCK:
        _TRACE_STATE[message_id] = {
            "chat_id": chat_id,
            "content": (content or "").replace("\r", " ").replace("\n", " ").strip(),
            "nodes": [],
        }


def trace_node(*, message_id: str, node_name: str) -> None:
    with _TRACE_LOCK:
        state = _TRACE_STATE.get(message_id)
        if not state:
            return
        nodes = state.get("nodes")
        if isinstance(nodes, list):
            nodes.append(node_name)


def trace_finish(*, message_id: str, suffix_status: str = "ok") -> None:
    with _TRACE_LOCK:
        state = _TRACE_STATE.pop(message_id, None)
    if not state:
        return
    content = str(state.get("content", ""))
    nodes = state.get("nodes", [])
    if not isinstance(nodes, list):
        nodes = []
    if suffix_status and suffix_status != "ok" and nodes:
        nodes[-1] = f"{nodes[-1]}({suffix_status})"
    path = "→".join(nodes) if nodes else "无节点"
    line = f"[ {content} ] : {path}"
    LOGGER.info(line)
    print(line, flush=True)


__all__ = ["trace_start", "trace_node", "trace_finish"]
