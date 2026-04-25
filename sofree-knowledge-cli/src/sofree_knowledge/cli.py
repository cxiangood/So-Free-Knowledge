from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .archive import collect_messages
from .config import load_env_file
from .feishu_client import FeishuClient
from .policy import KnowledgePolicyStore, VALID_SCOPES


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        result = args.func(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sofree-knowledge")
    parser.add_argument("--env-file", default="", help="Path to .env file.")
    parser.add_argument("--output-dir", default=".", help="Archive and policy root directory.")

    subparsers = parser.add_subparsers(dest="command")

    collect = subparsers.add_parser("collect-messages", help="Collect Feishu messages.")
    collect.add_argument("--chat-ids", default="", help="Comma-separated chat IDs, e.g. oc_xxx,oc_yyy.")
    collect.add_argument("--output-subdir", default="", help="Subdirectory under message_archive.")
    collect.add_argument("--include-visible-chats", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--start-time", default="", help="Start time, ISO date/time or Feishu timestamp.")
    collect.add_argument("--end-time", default="", help="End time, ISO date/time or Feishu timestamp.")
    collect.add_argument("--max-chats", type=int, default=1000)
    collect.add_argument("--max-messages-per-chat", type=int, default=1000)
    collect.add_argument("--page-size", type=int, default=50)
    collect.set_defaults(func=cmd_collect_messages)

    set_scope = subparsers.add_parser("set-knowledge-scope", help="Set per-chat knowledge scope.")
    set_scope.add_argument("chat_id")
    set_scope.add_argument("scope", choices=sorted(VALID_SCOPES))
    set_scope.set_defaults(func=cmd_set_knowledge_scope)

    get_scope = subparsers.add_parser("get-knowledge-scope", help="Get per-chat knowledge scope.")
    get_scope.add_argument("chat_id")
    get_scope.set_defaults(func=cmd_get_knowledge_scope)

    return parser


def cmd_collect_messages(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    manifest = collect_messages(
        client=FeishuClient(),
        output_dir=args.output_dir,
        output_subdir=args.output_subdir,
        chat_ids=args.chat_ids or None,
        include_visible_chats=args.include_visible_chats,
        start_time=args.start_time,
        end_time=args.end_time,
        max_chats=args.max_chats,
        max_messages_per_chat=args.max_messages_per_chat,
        page_size=args.page_size,
    )
    manifest["ok"] = True
    return manifest


def cmd_set_knowledge_scope(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    result = KnowledgePolicyStore(args.output_dir).set_scope(args.chat_id, args.scope)
    result["ok"] = True
    return result


def cmd_get_knowledge_scope(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    result = KnowledgePolicyStore(args.output_dir).get_scope(args.chat_id)
    result["ok"] = True
    return result


def prepare_env(args: argparse.Namespace) -> None:
    env_file = args.env_file
    if not env_file:
        candidate = Path(args.output_dir).expanduser() / "So-Free-Knowledge" / ".env"
        env_file = str(candidate) if candidate.exists() else ""
    load_env_file(env_file)


if __name__ == "__main__":
    raise SystemExit(main())
