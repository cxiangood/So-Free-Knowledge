from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..shared.models import LiftedCard
from ..shared.utils import now_utc_iso
from ..store.io import append_jsonl
from .feishu import load_env_file, resolve_sender_credentials


@dataclass(slots=True)
class TaskPushConfig:
    enabled: bool = False
    chat_id: str = ""
    env_file: str = ""


@dataclass(slots=True)
class TaskPushAttempt:
    task_id: str
    run_id: str
    chat_id: str
    card_payload: dict[str, Any]
    status: str
    message_id: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "chat_id": self.chat_id,
            "card_payload": self.card_payload,
            "status": self.status,
            "message_id": self.message_id,
            "error": self.error,
            "created_at": now_utc_iso(),
        }

    def to_pending_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "chat_id": self.chat_id,
            "card_payload": self.card_payload,
            "error": self.error,
            "retry_count": 0,
            "created_at": now_utc_iso(),
        }


@dataclass(slots=True)
class TextPushResult:
    chat_id: str
    status: str
    message_id: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "status": self.status,
            "message_id": self.message_id,
            "error": self.error,
            "created_at": now_utc_iso(),
        }


def _load_feishu_client_class():
    try:
        from sofree_knowledge.feishu_client import FeishuClient

        return FeishuClient
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        cli_src = repo_root / "sofree-knowledge-cli" / "src"
        cli_src_str = str(cli_src)
        if cli_src_str not in sys.path:
            sys.path.insert(0, cli_src_str)
        from sofree_knowledge.feishu_client import FeishuClient

        return FeishuClient


@contextmanager
def _temporary_feishu_credentials(app_id: str, app_secret: str):
    import os

    old_app_id = os.environ.get("FEISHU_APP_ID")
    old_app_secret = os.environ.get("FEISHU_APP_SECRET")
    os.environ["FEISHU_APP_ID"] = app_id
    os.environ["FEISHU_APP_SECRET"] = app_secret
    try:
        yield
    finally:
        if old_app_id is None:
            os.environ.pop("FEISHU_APP_ID", None)
        else:
            os.environ["FEISHU_APP_ID"] = old_app_id
        if old_app_secret is None:
            os.environ.pop("FEISHU_APP_SECRET", None)
        else:
            os.environ["FEISHU_APP_SECRET"] = old_app_secret


def build_task_card_payload(card: LiftedCard, *, run_id: str, task_id: str) -> dict[str, Any]:
    evidence = card.evidence[0] if card.evidence else ""
    tags = " ".join(f"`{tag}`" for tag in card.tags[:6]) if card.tags else ""
    markdown = f"""
        ### 🧠 关键摘要
        > **{card.summary}**

        ---

        ### 📌 详情信息

        | 项目 | 内容 |
        |------|------|
        | ⚠️ 问题 | {card.problem} |
        | 💡 建议 | {card.suggestion} |
        | 👥 相关人员 | {card.target_audience} |

        ---

        <sub>📎 相关片段：</sub>  
        <sub>> {evidence}</sub>
    """
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "red",
            "title": {"tag": "plain_text", "content": f"[Task] {card.title[:60]}"},
        },
        "elements": [{"tag": "markdown", "content": markdown}],
    }


def push_task_card(*, config: TaskPushConfig, run_id: str, task_id: str, card: LiftedCard) -> TaskPushAttempt:
    payload = build_task_card_payload(card, run_id=run_id, task_id=task_id)
    if not config.enabled:
        return TaskPushAttempt(task_id=task_id, run_id=run_id, chat_id=config.chat_id, card_payload=payload, status="skipped", error="task push disabled")
    chat_id = str(config.chat_id).strip()
    if not chat_id:
        return TaskPushAttempt(
            task_id=task_id,
            run_id=run_id,
            chat_id="",
            card_payload=payload,
            status="failed",
            error="task push chat_id is required when task push is enabled",
        )
    load_env_file(config.env_file)
    app_id, app_secret = resolve_sender_credentials()
    if not app_id or not app_secret:
        return TaskPushAttempt(
            task_id=task_id,
            run_id=run_id,
            chat_id=chat_id,
            card_payload=payload,
            status="failed",
            error="missing credentials: CARD_SENDER_APP_ID/CARD_SENDER_APP_SECRET or FEISHU_APP_ID/FEISHU_APP_SECRET",
        )
    FeishuClient = _load_feishu_client_class()
    try:
        with _temporary_feishu_credentials(app_id, app_secret):
            client = FeishuClient()
            sent = client.send_message(receive_id=chat_id, receive_id_type="chat_id", msg_type="interactive", content=payload)
    except Exception as exc:
        return TaskPushAttempt(
            task_id=task_id,
            run_id=run_id,
            chat_id=chat_id,
            card_payload=payload,
            status="failed",
            error=str(exc),
        )
    return TaskPushAttempt(
        task_id=task_id,
        run_id=run_id,
        chat_id=chat_id,
        card_payload=payload,
        status="sent",
        message_id=str(sent.get("message_id", "")),
        error="",
    )


def queue_failed_pushes(state_dir: str | Path, attempts: list[TaskPushAttempt]) -> int:
    rows = [item.to_pending_dict() for item in attempts if item.status == "failed"]
    if not rows:
        return 0
    append_jsonl(Path(state_dir) / "pending_task_push.jsonl", rows)
    return len(rows)


def push_text_message(*, chat_id: str, text: str, env_file: str = "") -> TextPushResult:
    target = str(chat_id or "").strip()
    if not target:
        return TextPushResult(chat_id="", status="failed", error="chat_id is required")
    body = str(text or "").strip()
    if not body:
        return TextPushResult(chat_id=target, status="failed", error="text body is empty")
    load_env_file(env_file)
    app_id, app_secret = resolve_sender_credentials()
    if not app_id or not app_secret:
        return TextPushResult(
            chat_id=target,
            status="failed",
            error="missing credentials: CARD_SENDER_APP_ID/CARD_SENDER_APP_SECRET or FEISHU_APP_ID/FEISHU_APP_SECRET",
        )
    FeishuClient = _load_feishu_client_class()
    try:
        with _temporary_feishu_credentials(app_id, app_secret):
            client = FeishuClient()
            sent = client.send_message(
                receive_id=target,
                receive_id_type="chat_id",
                msg_type="text",
                content={"text": body},
            )
    except Exception as exc:
        return TextPushResult(chat_id=target, status="failed", error=str(exc))
    return TextPushResult(chat_id=target, status="sent", message_id=str(sent.get("message_id", "")))


__all__ = [
    "TaskPushConfig",
    "TaskPushAttempt",
    "TextPushResult",
    "build_task_card_payload",
    "push_task_card",
    "push_text_message",
    "queue_failed_pushes",
]

if __name__ == "__main__":
    import json

    sample_card = LiftedCard(
        card_id="card-123",
        candidate_id="candidate-123",
        title="示例卡片标题",
        summary="这是一个示例卡片的摘要信息，用于展示推送功能。",
        problem="存在一个未决问题，需要明确答复或行动。",
        suggestion="建议将该信号转为待办并指定负责人。",
        target_audience="张三、李四",
        evidence=["这是相关的证据片段。"],
        tags=["tag1", "tag2"],
        confidence=0.85,
        suggested_target=None,
        source_message_ids=["msg-123"],
    )
    attempt = push_task_card(
        config=TaskPushConfig(enabled=True, chat_id="oc_9663f97db577d40181f3ccc9a4ef4b03", env_file=".env"),
        run_id="run-001",
        task_id="task-001",
        card=sample_card,
    )
    print(json.dumps(attempt.to_dict(), ensure_ascii=False, indent=2))