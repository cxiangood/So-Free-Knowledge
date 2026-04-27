from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .shared_types import LiftedCard, now_utc_iso


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


def _load_simple_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _ensure_env_loaded(env_file: str) -> None:
    if env_file.strip():
        _load_simple_env(Path(env_file).expanduser())
        return
    default_path = Path.cwd() / ".env"
    _load_simple_env(default_path)


def _resolve_sender_credentials() -> tuple[str, str]:
    app_id = (
        str(os.getenv("CARD_SENDER_APP_ID", "")).strip()
        or str(os.getenv("FEISHU_APP_ID", "")).strip()
    )
    app_secret = (
        str(os.getenv("CARD_SENDER_APP_SECRET", "")).strip()
        or str(os.getenv("FEISHU_APP_SECRET", "")).strip()
    )
    return app_id, app_secret


@contextmanager
def _temporary_feishu_credentials(app_id: str, app_secret: str):
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


def _load_feishu_client_class():
    try:
        from sofree_knowledge.feishu_client import FeishuClient

        return FeishuClient
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        cli_src = repo_root / "sofree-knowledge-cli" / "src"
        cli_src_str = str(cli_src)
        if cli_src_str not in sys.path:
            sys.path.insert(0, cli_src_str)
        from sofree_knowledge.feishu_client import FeishuClient

        return FeishuClient


def build_task_card_payload(card: LiftedCard, *, run_id: str, task_id: str) -> dict[str, Any]:
    evidence = card.evidence[0] if card.evidence else ""
    tags = " ".join(f"`{tag}`" for tag in card.tags[:6]) if card.tags else ""
    markdown = (
        f"**{card.summary}**\n"
        f"- 建议动作: {card.suggestion}\n"
        f"- 置信度: {card.confidence:.2f}\n"
        f"- 证据片段: {evidence}\n"
        f"- 标签: {tags or '无'}\n"
        f"- task_id: `{task_id}`\n"
        f"- run_id: `{run_id}`"
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "red",
            "title": {"tag": "plain_text", "content": f"[Task] {card.title[:60]}"},
        },
        "elements": [{"tag": "markdown", "content": markdown}],
    }


def push_task_card(
    *,
    config: TaskPushConfig,
    run_id: str,
    task_id: str,
    card: LiftedCard,
) -> TaskPushAttempt:
    payload = build_task_card_payload(card, run_id=run_id, task_id=task_id)
    if not config.enabled:
        return TaskPushAttempt(
            task_id=task_id,
            run_id=run_id,
            chat_id=config.chat_id,
            card_payload=payload,
            status="skipped",
            error="task push disabled",
        )
    chat_id = str(config.chat_id).strip()
    if not chat_id:
        return TaskPushAttempt(
            task_id=task_id,
            run_id=run_id,
            chat_id=chat_id,
            card_payload=payload,
            status="failed",
            error="task push chat_id is required when task push is enabled",
        )

    _ensure_env_loaded(config.env_file)
    app_id, app_secret = _resolve_sender_credentials()
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
            sent = client.send_message(
                receive_id=chat_id,
                receive_id_type="chat_id",
                msg_type="interactive",
                content=payload,
            )
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

