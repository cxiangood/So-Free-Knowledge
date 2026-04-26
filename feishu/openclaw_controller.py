from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feishu.apis import BotProfile, FeishuBotGateway
from utils import getenv


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_message(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.message:
        return args.message
    raise SystemExit("Provide --message or --file.")


def resolve_config(
    app_id: str | None = None,
    app_secret: str | None = None,
    chat_id: str | None = None,
) -> tuple[str, str, str]:
    load_env_file(Path(__file__).resolve().parents[1] / ".env")
    resolved_app_id = app_id or getenv("SOFREE_FEISHU_APP_ID") or getenv("FEISHU_APP_ID")
    resolved_app_secret = app_secret or getenv("SOFREE_FEISHU_APP_SECRET") or getenv("FEISHU_APP_SECRET")
    resolved_chat_id = chat_id or getenv("OPENCLAW_CHAT_ID")

    missing = [
        name
        for name, value in {
            "app-id": resolved_app_id,
            "app-secret": resolved_app_secret,
            "chat-id": resolved_chat_id,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required values: {', '.join(missing)}")
    return resolved_app_id, resolved_app_secret, resolved_chat_id


def send_openclaw_message(
    message: str,
    *,
    app_id: str | None = None,
    app_secret: str | None = None,
    chat_id: str | None = None,
    receive_id_type: str = "chat_id",
) -> str:
    resolved_app_id, resolved_app_secret, resolved_chat_id = resolve_config(app_id, app_secret, chat_id)
    profile = BotProfile(
        bot_id="sofree-controller",
        display_name="SoFree Controller",
        role_name="controller",
        role_prompt="Send structured prompts to OpenClaw through Feishu.",
        app_id=resolved_app_id,
        app_secret=resolved_app_secret,
    )
    gateway = FeishuBotGateway([profile])
    return gateway.send_text(profile.bot_id, resolved_chat_id, message, receive_id_type=receive_id_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a prompt to a Feishu chat that contains the OpenClaw bot.")
    parser.add_argument("--app-id", default=getenv("SOFREE_FEISHU_APP_ID") or getenv("FEISHU_APP_ID"))
    parser.add_argument("--app-secret", default=getenv("SOFREE_FEISHU_APP_SECRET") or getenv("FEISHU_APP_SECRET"))
    parser.add_argument("--chat-id", default=getenv("OPENCLAW_CHAT_ID"), help="Feishu chat_id, usually starts with oc_.")
    parser.add_argument("--receive-id-type", default="chat_id", choices=["chat_id", "open_id", "user_id", "union_id"])
    parser.add_argument("--message", help="Message text to send.")
    parser.add_argument("--file", help="UTF-8 text/markdown file to send as the message body.")
    args = parser.parse_args()
    message_id = send_openclaw_message(
        read_message(args),
        app_id=args.app_id,
        app_secret=args.app_secret,
        chat_id=args.chat_id,
        receive_id_type=args.receive_id_type,
    )
    print(message_id)


if __name__ == "__main__":
    main()
