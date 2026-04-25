from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .archive import collect_messages
from .auth import (
    DEFAULT_REDIRECT_URI,
    DEFAULT_SCOPE,
    auth_status,
    build_authorization_url,
    exchange_code_for_token,
    init_token,
)
from .config import load_env_file, resolve_env_file
from .feishu_client import FeishuClient
from .lingo_context import (
    build_lingo_judge_prompt,
    extract_keyword_contexts,
    parse_lingo_judgements,
    publishable_lingo_judgements,
)
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

    auth_url = subparsers.add_parser("auth-url", help="Print Feishu OAuth authorization URL.")
    auth_url.add_argument("--redirect-uri", default=DEFAULT_REDIRECT_URI)
    auth_url.add_argument("--scope", default=DEFAULT_SCOPE)
    auth_url.set_defaults(func=cmd_auth_url)

    init = subparsers.add_parser("init-token", help="Start Feishu OAuth flow.")
    init.add_argument("--redirect-uri", default=DEFAULT_REDIRECT_URI)
    init.add_argument("--scope", default=DEFAULT_SCOPE)
    init.add_argument("--token-file", default="")
    init.add_argument("--enable-autofill", action="store_true")
    init.set_defaults(func=cmd_init_token)

    exchange = subparsers.add_parser("exchange-code", help="Exchange OAuth code or redirect URL for token.")
    exchange.add_argument("code_or_url")
    exchange.add_argument("--token-file", default="")
    exchange.set_defaults(func=cmd_exchange_code)

    status = subparsers.add_parser("auth-status", help="Show local Feishu token status.")
    status.add_argument("--token-file", default="")
    status.set_defaults(func=cmd_auth_status)

    # Lingo commands
    lingo = subparsers.add_parser("lingo", help="Lingo (glossary) operations.")
    lingo_subparsers = lingo.add_subparsers(dest="lingo_command")

    extract_contexts = lingo_subparsers.add_parser("extract-contexts", help="Extract keyword contexts from messages.")
    extract_contexts.add_argument("--keywords", required=True, help="Comma-separated keywords to search for.")
    extract_contexts.add_argument("--messages-file", required=True, help="JSON file containing messages array.")
    extract_contexts.add_argument("--before", type=int, default=1, help="Number of messages before match to include.")
    extract_contexts.add_argument("--after", type=int, default=1, help="Number of messages after match to include.")
    extract_contexts.add_argument("--max-contexts", type=int, default=30, help="Maximum number of contexts to return.")
    extract_contexts.set_defaults(func=cmd_lingo_extract_contexts)

    build_prompt = lingo_subparsers.add_parser("build-judge-prompt", help="Build LLM prompt for lingo judgement.")
    build_prompt.add_argument("--keywords", required=True, help="Comma-separated keywords.")
    build_prompt.add_argument("--contexts-file", required=True, help="JSON file containing contexts array from extract-contexts.")
    build_prompt.set_defaults(func=cmd_lingo_build_judge_prompt)

    parse_judgements = lingo_subparsers.add_parser("parse-judgements", help="Parse LLM judgement output.")
    parse_judgements.add_argument("--judgements-file", required=True, help="File containing raw LLM output (JSON or text with code fences).")
    parse_judgements.add_argument("--publishable-only", action="store_true", help="Only return publishable judgements (key/black types with value).")
    parse_judgements.set_defaults(func=cmd_lingo_parse_judgements)

    return parser


def cmd_collect_messages(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    result = collect_messages(
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
    result["ok"] = True
    return result


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


def cmd_auth_url(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    return {
        "ok": True,
        "authorization_url": build_authorization_url(
            redirect_uri=args.redirect_uri,
            scope=args.scope,
        ),
    }


def cmd_init_token(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    result = init_token(
        enable_autofill=args.enable_autofill,
        redirect_uri=args.redirect_uri,
        scope=args.scope,
        token_file=args.token_file or None,
    )
    result["ok"] = True
    return result


def cmd_exchange_code(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    result = exchange_code_for_token(args.code_or_url, token_file=args.token_file or None)
    result["ok"] = True
    return result


def cmd_auth_status(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    result = auth_status(token_file=args.token_file or None)
    result["ok"] = True
    return result


def prepare_env(args: argparse.Namespace) -> None:
    env_file = resolve_env_file(args.env_file, output_dir=args.output_dir)
    load_env_file(env_file)


def cmd_lingo_extract_contexts(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    with open(args.messages_file, "r", encoding="utf-8") as f:
        messages = json.load(f)
    contexts = extract_keyword_contexts(
        keywords=keywords,
        messages=messages,
        before=args.before,
        after=args.after,
        max_contexts=args.max_contexts,
    )
    return {
        "ok": True,
        "keywords": keywords,
        "contexts": contexts,
        "count": len(contexts),
    }


def cmd_lingo_build_judge_prompt(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    with open(args.contexts_file, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    prompt = build_lingo_judge_prompt(
        keywords=keywords,
        contexts=contexts,
    )
    return {
        "ok": True,
        "prompt": prompt,
    }


def cmd_lingo_parse_judgements(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    with open(args.judgements_file, "r", encoding="utf-8") as f:
        raw_judgements = f.read()
    judgements = parse_lingo_judgements(raw_judgements)
    if args.publishable_only:
        judgements = publishable_lingo_judgements(judgements)
    return {
        "ok": True,
        "judgements": judgements,
        "count": len(judgements),
    }


if __name__ == "__main__":
    raise SystemExit(main())
