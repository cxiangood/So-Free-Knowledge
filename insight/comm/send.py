from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any
from utils import getenv

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
    details: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "chat_id": self.chat_id,
            "card_payload": self.card_payload,
            "status": self.status,
            "message_id": self.message_id,
            "error": self.error,
            "details": list(self.details or []),
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
    receive_id_type: str = "chat_id"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "status": self.status,
            "message_id": self.message_id,
            "error": self.error,
            "receive_id_type": self.receive_id_type,
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

    old_app_id = getenv("FEISHU_APP_ID")
    old_app_secret = getenv("FEISHU_APP_SECRET")
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
    tags = " ".join(f"{tag}" for tag in card.tags[:6]) if card.tags else ""

    elements = [
        # 关键摘要模块
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**{card.summary}**"
            }
        },
        # 分割线
        {"tag": "hr"},
        # 详情信息模块
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "**📌 详情信息**"
            }
        },
    ]

    # 标签模块（如果有标签）
    # if tags:
    #     elements.extend([
    #         {
    #             "tag": "div",
    #             "fields": [
    #                 {
    #                     "is_short": False,
    #                     "text": {
    #                         "tag": "lark_md",
    #                         "content": "**🏷️ 标签**{}".format(tags)
    #                     }
    #                 }
    #             ]
    #         }
    #     ])
    if card.times:
        elements.append(
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": "default",
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": "🕒 时间"
                                }
                            }
                        ]
                    },
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 4,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": card.times
                                }
                            }
                        ]
                    }
                ]
            },
        )
    if card.locations:
        elements.append(
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": "default",
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": "📍 地点"
                                }
                            }
                        ]
                    },
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 4,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": card.locations
                                }
                            }
                        ]
                    }
                ]
            },
        )
    if card.participants:
        elements.append(
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": "default",
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": "👥 人员"
                                }
                            }
                        ]
                    },
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 4,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": "、".join(card.participants)
                                }
                            }
                        ]
                    }
                ]
            },
        )
    if card.suggestion:
        elements.append(
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "none",
                "horizontal_spacing": "default",
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": "💡 建议"
                                }
                            }
                        ]
                    },
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 4,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": card.suggestion
                                }
                            }
                        ]
                    }
                ]
            },
        )
    if card.problem:
        elements.append(
            {
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "horizontal_spacing": "default",
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": "⚠️ 问题"
                                }
                            }
                        ]
                    },
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 4,
                        "vertical_align": "middle",
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "plain_text",
                                    "content": card.problem
                                }
                            }
                        ]
                    }
                ]
            },
        )
    # 相关片段模块（如果有证据）
    if evidence:
        elements.extend([
            {"tag": "hr"},
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"📎 相关片段：{evidence}"
                    }
                ]
            }
        ])

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": f"[Task] {card.title[:60]}"},
        },
        "elements": elements
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


def push_message_by_receive_id(
    *,
    receive_id: str,
    receive_id_type: str,
    msg_type: str,
    content: dict[str, Any] | str,
    env_file: str = "",
) -> TextPushResult:
    target = str(receive_id or "").strip()
    rid_type = str(receive_id_type or "").strip() or "chat_id"
    if not target:
        return TextPushResult(chat_id="", status="failed", error="receive_id is required", receive_id_type=rid_type)
    load_env_file(env_file)
    app_id, app_secret = resolve_sender_credentials()
    if not app_id or not app_secret:
        return TextPushResult(
            chat_id=target,
            status="failed",
            error="missing credentials: CARD_SENDER_APP_ID/CARD_SENDER_APP_SECRET or FEISHU_APP_ID/FEISHU_APP_SECRET",
            receive_id_type=rid_type,
        )
    FeishuClient = _load_feishu_client_class()
    try:
        with _temporary_feishu_credentials(app_id, app_secret):
            client = FeishuClient()
            sent = client.send_message(
                receive_id=target,
                receive_id_type=rid_type,
                msg_type=msg_type,
                content=content,
            )
    except Exception as exc:
        return TextPushResult(chat_id=target, status="failed", error=str(exc), receive_id_type=rid_type)
    return TextPushResult(chat_id=target, status="sent", message_id=str(sent.get("message_id", "")), receive_id_type=rid_type)


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
    return push_message_by_receive_id(
        receive_id=target,
        receive_id_type="chat_id",
        msg_type="text",
        content={"text": body},
        env_file=env_file,
    )


def push_task_card_to_user_targets(
    *,
    run_id: str,
    task_id: str,
    card: LiftedCard,
    env_file: str,
    targets: list[tuple[str, str, str]],
    fallback_receive_id: str,
    fallback_receive_id_type: str,
) -> TaskPushAttempt:
    payload = build_task_card_payload(card, run_id=run_id, task_id=task_id)
    details: list[dict[str, Any]] = []
    any_sent = False
    any_failed = False
    first_message_id = ""
    failed_rows: list[dict[str, str]] = []
    target_receive_id_type = (getenv("TASK_PUSH_RECEIVE_ID_TYPE", "user_id") or "user_id").strip()
    for audience_name, receive_id, resolve_error_code in targets:
        if receive_id:
            sent = push_message_by_receive_id(
                receive_id=receive_id,
                receive_id_type=target_receive_id_type,
                msg_type="interactive",
                content=payload,
                env_file=env_file,
            )
            error_code = _extract_error_code(sent.error)
            details.append(
                {
                    "name": audience_name,
                    "receive_id": receive_id,
                    "receive_id_type": sent.receive_id_type,
                    "status": sent.status,
                    "message_id": sent.message_id,
                    "error": sent.error,
                    "error_code": error_code,
                    "fallback": False,
                }
            )
            if sent.status == "sent":
                any_sent = True
                if not first_message_id:
                    first_message_id = sent.message_id
                continue
            any_failed = True
            failed_rows.append({"name": audience_name, "error_code": error_code or "unknown"})
        else:
            any_failed = True
            failed_rows.append({"name": audience_name, "error_code": resolve_error_code or "resolve_failed"})

    fallback_text_result = TextPushResult(chat_id=fallback_receive_id, status="skipped", receive_id_type=fallback_receive_id_type)
    fallback_card_result = TextPushResult(chat_id=fallback_receive_id, status="skipped", receive_id_type=fallback_receive_id_type)
    if failed_rows and fallback_receive_id:
        lines = ["任务卡目标发送失败，已统一回退。", "失败明细："]
        for row in failed_rows:
            lines.append(f"- {row['name']}: {row['error_code']}")
        fallback_text_result = push_message_by_receive_id(
            receive_id=fallback_receive_id,
            receive_id_type=fallback_receive_id_type,
            msg_type="text",
            content={"text": "\n".join(lines)},
            env_file=env_file,
        )
        details.append(
            {
                "name": "fallback_notice",
                "receive_id": fallback_receive_id,
                "receive_id_type": fallback_receive_id_type,
                "status": fallback_text_result.status,
                "message_id": fallback_text_result.message_id,
                "error": fallback_text_result.error,
                "error_code": _extract_error_code(fallback_text_result.error),
                "fallback": True,
            }
        )
        fallback_card_result = push_message_by_receive_id(
            receive_id=fallback_receive_id,
            receive_id_type=fallback_receive_id_type,
            msg_type="interactive",
            content=payload,
            env_file=env_file,
        )
        details.append(
            {
                "name": "fallback_card",
                "receive_id": fallback_receive_id,
                "receive_id_type": fallback_receive_id_type,
                "status": fallback_card_result.status,
                "message_id": fallback_card_result.message_id,
                "error": fallback_card_result.error,
                "error_code": _extract_error_code(fallback_card_result.error),
                "fallback": True,
            }
        )
        if fallback_card_result.status == "sent":
            any_sent = True
            if not first_message_id:
                first_message_id = fallback_card_result.message_id
    elif failed_rows and not fallback_receive_id:
        details.append(
            {
                "name": "fallback_missing",
                "receive_id": "",
                "receive_id_type": fallback_receive_id_type,
                "status": "failed",
                "message_id": "",
                "error": "missing fallback receive id",
                "error_code": "fallback_missing",
                "fallback": True,
            }
        )
    status = "failed"
    if any_sent and not any_failed:
        status = "sent"
    elif any_sent and any_failed:
        status = "partial"
    error = ""
    if status != "sent":
        error = "; ".join(item.get("error", "") for item in details if item.get("error"))
    return TaskPushAttempt(
        task_id=task_id,
        run_id=run_id,
        chat_id="",
        card_payload=payload,
        status=status,
        message_id=first_message_id,
        error=error,
        details=details,
    )


def _extract_error_code(error: str) -> str:
    text = str(error or "")
    if not text:
        return ""
    m = re.search(r"code=([0-9]+)", text)
    if m:
        return m.group(1)
    return ""


__all__ = [
    "TaskPushConfig",
    "TaskPushAttempt",
    "TextPushResult",
    "build_task_card_payload",
    "push_task_card",
    "push_message_by_receive_id",
    "push_task_card_to_user_targets",
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
