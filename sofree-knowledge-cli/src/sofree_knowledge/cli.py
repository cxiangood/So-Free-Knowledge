from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .archive import collect_messages
from .assistant.profile import (
    assistant_profile_default_path,
    build_profile_review_card,
    build_profile_overrides,
    build_retrieval_overrides,
    build_schedule_overrides,
    load_assistant_profile_config,
    save_assistant_profile_config,
    suggest_profile_from_online_inputs,
)
from .assistant.service import build_personal_brief_command_result, recommend_command_result, resolve_push_target
from .assistant.training import export_dual_tower_samples, load_dual_tower_samples, train_dual_tower_baseline
from .auth import (
    DEFAULT_REDIRECT_URI,
    DEFAULT_SCOPE,
    auth_status,
    build_authorization_url,
    device_login,
    ensure_user_auth,
    exchange_code_for_token,
    init_token,
    resume_device_login,
    start_device_login,
)
from .config import get_user_access_token, get_user_identity, load_env_file, resolve_env_file
from .confused_detector import (
    build_confused_judge_prompt,
    detect_confused_candidates,
    format_inline_explanation,
    parse_confused_judgement,
)
from .feishu_client import FeishuClient
from .assistant_brief import build_personal_brief
from .assistant_online import collect_online_personal_inputs
from .lingo_context import (
    build_lingo_judge_prompt,
    extract_keyword_contexts,
    parse_lingo_judgements,
    publishable_lingo_judgements,
)
from .lingo_auto import (
    parse_lingo_ai_review_judgements,
    run_lingo_auto_pipeline,
    sync_ai_review_judgements,
)
from .lingo_store import LingoStore
from .logging_config import configure_logging, get_logger
from .policy import KnowledgePolicyStore, VALID_SCOPES
from . import wikisheet as wikisheet_module


LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        configure_logging(
            level=args.log_level,
            log_file=args.log_file,
            app_name="SOFREE-CLI",
            quiet=args.quiet,
            force=True,
        )
        LOGGER.debug("running command: %s", args.command)
        result = args.func(args)
    except Exception as exc:
        if logging.getLogger().handlers:
            LOGGER.exception("command failed")
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sofree-knowledge")
    parser.add_argument("--env-file", default="", help="Path to .env file.")
    parser.add_argument("--output-dir", default=".", help="Archive and policy root directory.")
    parser.add_argument("--token-file", default="", help="Optional user token file path.")
    parser.add_argument("--user-open-id", default="", help="Optional user scope. Runtime state will be isolated under output-dir/users/<open_id>.")
    parser.add_argument("--log-level", default="", help="Logging level. Defaults to SOFREE_LOG_LEVEL or INFO.")
    parser.add_argument("--log-file", default="", help="Optional path for persistent logs. Defaults to SOFREE_LOG_FILE.")
    parser.add_argument("--quiet", action="store_true", help="Disable terminal logs; file logs still work when --log-file is set.")
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

    auth = subparsers.add_parser("auth", help="Authentication operations.")
    auth_subparsers = auth.add_subparsers(dest="auth_command")

    auth_login = auth_subparsers.add_parser("login", help="Run Feishu device-flow login.")
    auth_login.add_argument("--scope", default=DEFAULT_SCOPE)
    auth_login.add_argument("--token-file", default="")
    auth_login.add_argument("--no-browser", action="store_true", help="Do not open the verification URL automatically.")
    auth_login.add_argument("--no-wait", action="store_true", help="Start device flow and return device_code immediately.")
    auth_login.add_argument("--device-code", default="", help="Resume a previous device-flow login with a device code.")
    auth_login.add_argument("--interval", type=int, default=5, help="Polling interval used with --device-code.")
    auth_login.add_argument("--expires-in", type=int, default=240, help="Polling timeout used with --device-code.")
    auth_login.set_defaults(func=cmd_auth_login)

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

    build_prompt = lingo_subparsers.add_parser("build-judge-prompt", help="Build an AI review prompt for lingo judgement.")
    build_prompt.add_argument("--keywords", required=True, help="Comma-separated keywords.")
    build_prompt.add_argument("--contexts-file", required=True, help="JSON file containing contexts array from extract-contexts.")
    build_prompt.set_defaults(func=cmd_lingo_build_judge_prompt)

    parse_judgements = lingo_subparsers.add_parser("parse-judgements", help="Parse AI review judgement output.")
    parse_judgements.add_argument("--judgements-file", required=True, help="File containing raw AI review output (JSON or text with code fences).")
    parse_judgements.add_argument("--publishable-only", action="store_true", help="Only return publishable judgements (key/black types with value).")
    parse_judgements.set_defaults(func=cmd_lingo_parse_judgements)

    lingo_upsert = lingo_subparsers.add_parser("upsert", help="Create/update one lingo entry (remote by default).")
    lingo_upsert.add_argument("--keyword", required=True)
    lingo_upsert.add_argument("--type", required=True, choices=["key", "black", "confused", "nothing"])
    lingo_upsert.add_argument("--value", default="")
    lingo_upsert.add_argument("--aliases", default="", help="Comma-separated aliases.")
    lingo_upsert.add_argument("--source", default="manual")
    lingo_upsert.add_argument("--entity-id", default="")
    lingo_upsert.add_argument("--replace-entity-id", default="", help="Delete this remote entity id first, then create new one.")
    lingo_upsert.add_argument("--force-remote-create", action="store_true", help="Force remote create even if local mirror indicates already created.")
    lingo_upsert.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True, help="Write to Feishu Lingo remotely.")
    lingo_upsert.add_argument("--write-local", action=argparse.BooleanOptionalAction, default=True, help="Mirror to local lingo_entries.json.")
    lingo_upsert.add_argument("--context-ids", default="", help="Comma-separated context ids.")
    lingo_upsert.set_defaults(func=cmd_lingo_upsert)

    lingo_delete = lingo_subparsers.add_parser("delete", help="Delete lingo entry (remote by default, local optional).")
    lingo_delete.add_argument("--keyword", default="", help="Keyword for local delete.")
    lingo_delete.add_argument("--entity-id", default="", help="Remote entity id for Feishu delete.")
    lingo_delete.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True, help="Delete in Feishu Lingo remotely.")
    lingo_delete.add_argument("--delete-local", action=argparse.BooleanOptionalAction, default=True, help="Also delete local mirror by keyword if provided.")
    lingo_delete.set_defaults(func=cmd_lingo_delete)

    lingo_list = lingo_subparsers.add_parser("list", help="List lingo entries in local store.")
    lingo_list.add_argument("--limit", type=int, default=200)
    lingo_list.set_defaults(func=cmd_lingo_list)

    lingo_sync = lingo_subparsers.add_parser("sync-from-file", help="Sync lingo entries from judgements or entries JSON file into local store.")
    lingo_sync.add_argument("--input-file", required=True, help="JSON/text file. Supports raw judgements or {'items': [...]} format.")
    lingo_sync.add_argument("--publishable-only", action="store_true", help="Only keep key/black with non-empty value.")
    lingo_sync.add_argument("--source", default="sync")
    lingo_sync.add_argument("--force-remote-create", action="store_true", help="Force remote create for every item.")
    lingo_sync.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True, help="Write each entry to Feishu Lingo remotely.")
    lingo_sync.add_argument("--write-local", action=argparse.BooleanOptionalAction, default=True, help="Mirror synced entries to local store.")
    lingo_sync.set_defaults(func=cmd_lingo_sync_from_file)

    lingo_auto = lingo_subparsers.add_parser("auto-sync", help="Collect recent messages, mine candidate terms, then emit an AI review prompt and optionally sync reviewed results.")
    lingo_auto.add_argument("--recent-days", type=int, default=7, help="How many recent days of chat history to scan.")
    lingo_auto.add_argument("--min-run-interval-days", type=int, default=7, help="Minimum interval between successful auto-sync runs.")
    lingo_auto.add_argument("--force", action="store_true", help="Ignore last-run interval guard.")
    lingo_auto.add_argument("--chat-ids", default="", help="Comma-separated chat IDs. Empty means rely on visible chats.")
    lingo_auto.add_argument("--include-visible-chats", action=argparse.BooleanOptionalAction, default=True)
    lingo_auto.add_argument("--start-time", default="", help="Optional explicit start time override.")
    lingo_auto.add_argument("--end-time", default="", help="Optional explicit end time override.")
    lingo_auto.add_argument("--max-chats", type=int, default=200)
    lingo_auto.add_argument("--max-messages-per-chat", type=int, default=500)
    lingo_auto.add_argument("--page-size", type=int, default=50)
    lingo_auto.add_argument("--top-keywords", type=int, default=30)
    lingo_auto.add_argument("--candidate-limit", type=int, default=20)
    lingo_auto.add_argument("--min-frequency", type=int, default=2)
    lingo_auto.add_argument("--min-contexts", type=int, default=1)
    lingo_auto.add_argument("--context-before", type=int, default=1)
    lingo_auto.add_argument("--context-after", type=int, default=1)
    lingo_auto.add_argument("--max-contexts", type=int, default=80)
    lingo_auto.add_argument("--analyzer-enabled", action=argparse.BooleanOptionalAction, default=True)
    lingo_auto.add_argument("--judgements-file", default="", help="Optional AI review result JSON file. If omitted, only candidate mining + prompt generation are performed.")
    lingo_auto.add_argument("--publishable-only", action="store_true", help="When --judgements-file is provided, only sync create/append decisions.")
    lingo_auto.add_argument("--source", default="lingo_auto")
    lingo_auto.add_argument("--force-remote-create", action="store_true")
    lingo_auto.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True, help="When --judgements-file is provided, write reviewed entries to Feishu Lingo remotely.")
    lingo_auto.add_argument("--write-local", action=argparse.BooleanOptionalAction, default=True, help="When --judgements-file is provided, mirror reviewed entries to local store.")
    lingo_auto.set_defaults(func=cmd_lingo_auto_sync)

    # Confused conversation detection commands
    confused = subparsers.add_parser("confused", help="Confused conversation detection helpers.")
    confused_subparsers = confused.add_subparsers(dest="confused_command")

    confused_detect = confused_subparsers.add_parser("detect-candidates", help="Detect confused candidates from messages with lightweight rules.")
    confused_detect.add_argument("--messages-file", required=True, help="JSON file containing messages array.")
    confused_detect.add_argument("--target-message-id", default="", help="Optional target message ID to filter on.")
    confused_detect.add_argument("--reactions-file", default="", help="Optional JSON file containing reaction events array.")
    confused_detect.add_argument("--confused-reaction-keys", default="", help="Comma-separated Feishu reaction keys that should be treated as confusion signals.")
    confused_detect.add_argument("--max-followup-gap", type=int, default=3, help="How many recent messages to inspect for follow-up confusion.")
    confused_detect.add_argument("--max-candidates", type=int, default=20, help="Maximum number of candidates to return.")
    confused_detect.set_defaults(func=cmd_confused_detect_candidates)

    confused_prompt = confused_subparsers.add_parser("build-judge-prompt", help="Build LLM prompt for one confused candidate.")
    confused_prompt.add_argument("--candidate-file", required=True, help="JSON file containing one candidate object.")
    confused_prompt.set_defaults(func=cmd_confused_build_judge_prompt)

    confused_parse = confused_subparsers.add_parser("parse-judgement", help="Parse LLM confused judgement output.")
    confused_parse.add_argument("--judgement-file", required=True, help="File containing raw LLM judgement output (JSON or fenced JSON).")
    confused_parse.set_defaults(func=cmd_confused_parse_judgement)

    assistant = subparsers.add_parser("assistant", help="Personal assistant operations.")
    assistant_subparsers = assistant.add_subparsers(dest="assistant_command")

    assistant_profile_set = assistant_subparsers.add_parser("set-profile", help="Save assistant profile + schedule config.")
    assistant_profile_set.add_argument("--profile-file", default="", help="Optional profile json path.")
    assistant_profile_set.add_argument("--persona", default="", help="User persona/avatar hint.")
    assistant_profile_set.add_argument("--role", default="", help="User role/profession.")
    assistant_profile_set.add_argument("--businesses", default="", help="Comma-separated business tracks, e.g. A增长,B交付.")
    assistant_profile_set.add_argument("--interests", default="", help="Comma-separated interest keywords for group digest.")
    assistant_profile_set.add_argument("--mode", choices=["scheduled", "manual", "hybrid"], default="scheduled")
    assistant_profile_set.add_argument("--timezone", default="Asia/Shanghai")
    assistant_profile_set.add_argument("--weekly-brief-cron", default="0 9 * * MON")
    assistant_profile_set.add_argument("--nightly-interest-cron", default="0 21 * * *")
    assistant_profile_set.add_argument("--weekly-enabled", action=argparse.BooleanOptionalAction, default=True)
    assistant_profile_set.add_argument("--nightly-enabled", action=argparse.BooleanOptionalAction, default=True)
    assistant_profile_set.add_argument("--dual-tower-enabled", action=argparse.BooleanOptionalAction, default=None)
    assistant_profile_set.add_argument("--dual-tower-model", default="")
    assistant_profile_set.add_argument("--dual-tower-model-file", default="")
    assistant_profile_set.add_argument("--dual-tower-top-k", type=int, default=None)
    assistant_profile_set.add_argument("--dual-tower-min-score", type=float, default=None)
    assistant_profile_set.set_defaults(func=cmd_assistant_set_profile)

    assistant_profile_get = assistant_subparsers.add_parser("get-profile", help="Read assistant profile + schedule config.")
    assistant_profile_get.add_argument("--profile-file", default="", help="Optional profile json path.")
    assistant_profile_get.set_defaults(func=cmd_assistant_get_profile)

    assistant_profile_confirm = assistant_subparsers.add_parser("confirm-profile", help="Confirm the current assistant profile suggestion.")
    assistant_profile_confirm.add_argument("--profile-file", default="", help="Optional profile json path.")
    assistant_profile_confirm.set_defaults(func=cmd_assistant_confirm_profile)

    assistant_build = assistant_subparsers.add_parser("build-personal-brief", help="Build personal brief from docs/access/messages/knowledge.")
    assistant_build.add_argument("--online", action="store_true", help="Collect data from Feishu APIs online in one command.")
    assistant_build.add_argument("--documents-file", default="", help="JSON file containing documents array. Required in offline mode.")
    assistant_build.add_argument("--access-records-file", default="", help="Optional JSON file containing access records array.")
    assistant_build.add_argument("--messages-file", default="", help="Optional JSON file containing group messages array.")
    assistant_build.add_argument("--knowledge-file", default="", help="Optional JSON file containing knowledge items array.")
    assistant_build.add_argument("--target-user-id", default="", help="Optional target user id for access filtering.")
    assistant_build.add_argument("--token-file", default="", help="Optional token file for auto-resolving current user id in online mode.")
    assistant_build.add_argument("--chat-ids", default="", help="Comma-separated chat IDs for online mode.")
    assistant_build.add_argument("--include-visible-chats", action=argparse.BooleanOptionalAction, default=True)
    assistant_build.add_argument("--max-chats", type=int, default=20)
    assistant_build.add_argument("--max-messages-per-chat", type=int, default=200)
    assistant_build.add_argument("--max-drive-docs", type=int, default=50)
    assistant_build.add_argument("--max-knowledge", type=int, default=30)
    assistant_build.add_argument("--recent-days", type=int, default=7, help="Recent window in days for online docs/messages.")
    assistant_build.add_argument("--max-docs", type=int, default=10, help="Maximum number of ranked documents.")
    assistant_build.add_argument("--max-related", type=int, default=5, help="Maximum number of related messages/knowledge per document.")
    assistant_build.add_argument("--max-interest-items", type=int, default=8, help="Maximum number of interest digest messages.")
    assistant_build.add_argument("--profile-file", default="", help="Optional profile json path.")
    assistant_build.add_argument("--persona", default="", help="Override persona/avatar for this run.")
    assistant_build.add_argument("--role", default="", help="Override role/profession for this run.")
    assistant_build.add_argument("--businesses", default="", help="Override business tracks, comma-separated.")
    assistant_build.add_argument("--interests", default="", help="Override interests, comma-separated.")
    assistant_build.add_argument("--mode", choices=["scheduled", "manual", "hybrid"], default="", help="Override schedule mode.")
    assistant_build.add_argument("--timezone", default="", help="Override schedule timezone.")
    assistant_build.add_argument("--weekly-brief-cron", default="", help="Override weekly brief cron.")
    assistant_build.add_argument("--nightly-interest-cron", default="", help="Override nightly digest cron.")
    assistant_build.add_argument("--weekly-enabled", action=argparse.BooleanOptionalAction, default=None)
    assistant_build.add_argument("--nightly-enabled", action=argparse.BooleanOptionalAction, default=None)
    assistant_build.add_argument("--dual-tower-enabled", action=argparse.BooleanOptionalAction, default=None)
    assistant_build.add_argument("--dual-tower-model", default="")
    assistant_build.add_argument("--dual-tower-model-file", default="")
    assistant_build.add_argument("--dual-tower-top-k", type=int, default=None)
    assistant_build.add_argument("--dual-tower-min-score", type=float, default=None)
    assistant_build.add_argument("--push", action="store_true", help="Push assistant result to Feishu.")
    assistant_build.add_argument("--push-interest-card", action=argparse.BooleanOptionalAction, default=True, help="Whether to push interest digest card.")
    assistant_build.add_argument("--push-summary-card", action=argparse.BooleanOptionalAction, default=True, help="Whether to push summary card.")
    assistant_build.add_argument("--receive-chat-id", default="", help="Explicit target chat_id for push. If set, push to group chat.")
    assistant_build.add_argument("--receive-open-id", default="", help="Explicit target open_id for push. Used when --receive-chat-id is empty.")
    assistant_build.add_argument(
        "--output-format",
        choices=["all", "json", "doc", "card"],
        default="all",
        help="Output format. 'doc' is deprecated and kept for compatibility.",
    )
    assistant_build.set_defaults(func=cmd_assistant_build_personal_brief)

    assistant_recommend = assistant_subparsers.add_parser(
        "recommend",
        help="One-command OpenClaw recommendation with auto dual-tower enable/disable.",
    )
    assistant_recommend.add_argument("--target-user-id", default="", help="Optional target user id for access filtering.")
    assistant_recommend.add_argument("--token-file", default="", help="Optional token file for auto-resolving current user id.")
    assistant_recommend.add_argument("--chat-ids", default="", help="Comma-separated chat IDs for online mode.")
    assistant_recommend.add_argument("--include-visible-chats", action=argparse.BooleanOptionalAction, default=True)
    assistant_recommend.add_argument("--max-chats", type=int, default=20)
    assistant_recommend.add_argument("--max-messages-per-chat", type=int, default=200)
    assistant_recommend.add_argument("--max-drive-docs", type=int, default=50)
    assistant_recommend.add_argument("--max-knowledge", type=int, default=30)
    assistant_recommend.add_argument("--recent-days", type=int, default=7)
    assistant_recommend.add_argument("--max-docs", type=int, default=10)
    assistant_recommend.add_argument("--max-related", type=int, default=5)
    assistant_recommend.add_argument("--max-interest-items", type=int, default=8)
    assistant_recommend.add_argument("--profile-file", default="", help="Optional profile json path.")
    assistant_recommend.add_argument("--persona", default="", help="Override persona/avatar for this run.")
    assistant_recommend.add_argument("--role", default="", help="Override role/profession for this run.")
    assistant_recommend.add_argument("--businesses", default="", help="Override business tracks, comma-separated.")
    assistant_recommend.add_argument("--interests", default="", help="Override interests, comma-separated.")
    assistant_recommend.add_argument("--mode", choices=["scheduled", "manual", "hybrid"], default="")
    assistant_recommend.add_argument("--timezone", default="")
    assistant_recommend.add_argument("--weekly-brief-cron", default="")
    assistant_recommend.add_argument("--nightly-interest-cron", default="")
    assistant_recommend.add_argument("--weekly-enabled", action=argparse.BooleanOptionalAction, default=None)
    assistant_recommend.add_argument("--nightly-enabled", action=argparse.BooleanOptionalAction, default=None)
    assistant_recommend.add_argument("--dual-tower-model", default="")
    assistant_recommend.add_argument("--dual-tower-model-file", default="")
    assistant_recommend.add_argument("--dual-tower-top-k", type=int, default=None)
    assistant_recommend.add_argument("--dual-tower-min-score", type=float, default=None)
    assistant_recommend.add_argument("--dual-tower-min-samples", type=int, default=20)
    assistant_recommend.add_argument("--push", action="store_true", help="Push assistant result to Feishu.")
    assistant_recommend.add_argument("--push-interest-card", action=argparse.BooleanOptionalAction, default=True)
    assistant_recommend.add_argument("--push-summary-card", action=argparse.BooleanOptionalAction, default=True)
    assistant_recommend.add_argument("--receive-chat-id", default="")
    assistant_recommend.add_argument("--receive-open-id", default="")
    assistant_recommend.add_argument("--output-format", choices=["all", "json", "doc", "card"], default="all")
    assistant_recommend.set_defaults(func=cmd_assistant_recommend)

    assistant_export_samples = assistant_subparsers.add_parser(
        "export-dual-tower-samples",
        help="Export weak-supervision samples for dual-tower training.",
    )
    assistant_export_samples.add_argument("--documents-file", required=True)
    assistant_export_samples.add_argument("--access-records-file", required=True)
    assistant_export_samples.add_argument("--messages-file", required=True)
    assistant_export_samples.add_argument("--target-user-id", default="")
    assistant_export_samples.add_argument("--profile-file", default="", help="Optional profile json path.")
    assistant_export_samples.add_argument("--persona", default="", help="Override persona/avatar for this run.")
    assistant_export_samples.add_argument("--role", default="", help="Override role/profession for this run.")
    assistant_export_samples.add_argument("--businesses", default="", help="Override business tracks, comma-separated.")
    assistant_export_samples.add_argument("--interests", default="", help="Override interests, comma-separated.")
    assistant_export_samples.add_argument("--output-file", default="", help="Optional JSONL file path.")
    assistant_export_samples.set_defaults(func=cmd_assistant_export_dual_tower_samples)

    assistant_train_dual_tower = assistant_subparsers.add_parser(
        "train-dual-tower",
        help="Train a lightweight dual-tower baseline from exported JSONL samples.",
    )
    assistant_train_dual_tower.add_argument("--samples-file", required=True, help="JSONL file from export-dual-tower-samples.")
    assistant_train_dual_tower.add_argument("--output-file", default="", help="Optional model output JSON path.")
    assistant_train_dual_tower.add_argument("--min-token-weight", type=float, default=0.0)
    assistant_train_dual_tower.set_defaults(func=cmd_assistant_train_dual_tower)

    # Wiki sheet commands
    wikisheet = subparsers.add_parser("wikisheet", help="Wiki sheet operations.")
    wikisheet_subparsers = wikisheet.add_subparsers(dest="wikisheet_command")

    ws_create = wikisheet_subparsers.add_parser("create-sheet", help="Create a sheet node in Wiki.")
    ws_create.add_argument("--title", required=True, help="Sheet title.")
    ws_create.add_argument("--space-id", default="", help="Wiki space id. Supports literal 'my_library'.")
    ws_create.add_argument("--parent-node-token", default="", help="Create under this parent wiki node token.")
    ws_create.add_argument(
        "--identity",
        choices=["user", "bot", "auto"],
        default="auto",
        help="Token identity for API calls (default: auto, prefer user token).",
    )
    ws_create.set_defaults(func=wikisheet_module.cmd_create_sheet)

    ws_append = wikisheet_subparsers.add_parser("append-data", help="Append rows to a sheet.")
    ws_append.add_argument("--spreadsheet-token", required=True, help="Spreadsheet token.")
    ws_append.add_argument("--range", required=True, help="Target range, e.g. <sheetId>!A1 or <sheetId>!A1:D10.")
    ws_append.add_argument("--values", required=True, help="2D JSON array.")
    ws_append.add_argument(
        "--identity",
        choices=["user", "bot", "auto"],
        default="auto",
        help="Token identity for API calls (default: auto, prefer user token).",
    )
    ws_append.set_defaults(func=wikisheet_module.cmd_append_data)

    ws_update = wikisheet_subparsers.add_parser("update-data", help="Overwrite values in a range.")
    ws_update.add_argument("--spreadsheet-token", required=True, help="Spreadsheet token.")
    ws_update.add_argument("--range", required=True, help="Target range, e.g. <sheetId>!A1:D10.")
    ws_update.add_argument("--values", required=True, help="2D JSON array.")
    ws_update.add_argument(
        "--identity",
        choices=["user", "bot", "auto"],
        default="auto",
        help="Token identity for API calls (default: auto, prefer user token).",
    )
    ws_update.set_defaults(func=wikisheet_module.cmd_update_data)

    ws_delete_data = wikisheet_subparsers.add_parser("delete-data", help="Delete data in a range (clear values).")
    ws_delete_data.add_argument("--spreadsheet-token", required=True, help="Spreadsheet token.")
    ws_delete_data.add_argument("--range", required=True, help="Target range, e.g. <sheetId>!A1:D10.")
    ws_delete_data.add_argument(
        "--identity",
        choices=["user", "bot", "auto"],
        default="auto",
        help="Token identity for API calls (default: auto, prefer user token).",
    )
    ws_delete_data.set_defaults(func=wikisheet_module.cmd_delete_data)

    ws_delete_sheet = wikisheet_subparsers.add_parser("delete-sheet", help="Delete the spreadsheet file.")
    ws_delete_sheet.add_argument("--spreadsheet-token", required=True, help="Spreadsheet token.")
    ws_delete_sheet.add_argument(
        "--identity",
        choices=["user", "bot", "auto"],
        default="auto",
        help="Token identity for API calls (default: auto, prefer user token).",
    )
    ws_delete_sheet.set_defaults(func=wikisheet_module.cmd_delete_sheet)

    shortcut_brief = subparsers.add_parser(
        "brief",
        aliases=["b"],
        help="Shortcut: collect online data and push the knowledge aggregation card with sane defaults.",
    )
    shortcut_brief.add_argument("--token-file", default="", help="Optional token file for auto-resolving current user id.")
    shortcut_brief.add_argument("--chat-ids", default="", help="Comma-separated chat IDs. Empty means rely on visible chats.")
    shortcut_brief.add_argument("--include-visible-chats", action=argparse.BooleanOptionalAction, default=True)
    shortcut_brief.add_argument("--recent-days", type=int, default=7)
    shortcut_brief.add_argument("--max-chats", type=int, default=20)
    shortcut_brief.add_argument("--max-messages-per-chat", type=int, default=200)
    shortcut_brief.add_argument("--max-drive-docs", type=int, default=50)
    shortcut_brief.add_argument("--max-knowledge", type=int, default=30)
    shortcut_brief.add_argument("--profile-file", default="", help="Optional profile json path.")
    shortcut_brief.add_argument("--persona", default="", help="Override persona/avatar for this run.")
    shortcut_brief.add_argument("--role", default="", help="Override role/profession for this run.")
    shortcut_brief.add_argument("--businesses", default="", help="Override business tracks, comma-separated.")
    shortcut_brief.add_argument("--interests", default="", help="Override interests, comma-separated.")
    shortcut_brief.add_argument("--receive-chat-id", default="", help="Explicit target chat_id for push. If set, push to group chat.")
    shortcut_brief.add_argument("--receive-open-id", default="", help="Explicit target open_id for push. Used when --receive-chat-id is empty.")
    shortcut_brief.add_argument("--output-format", choices=["all", "json", "card"], default="card")
    shortcut_brief.add_argument("--push-interest-card", action=argparse.BooleanOptionalAction, default=True)
    shortcut_brief.add_argument("--push-summary-card", action=argparse.BooleanOptionalAction, default=False)
    shortcut_brief.set_defaults(func=cmd_shortcut_brief)

    shortcut_lingo = subparsers.add_parser(
        "lingo-write",
        aliases=["lw"],
        help="Shortcut: mine glossary candidates and optionally sync reviewed entries with defaults tuned for Feishu Lingo.",
    )
    shortcut_lingo.add_argument("--token-file", default="", help="Optional token file for auth context.")
    shortcut_lingo.add_argument("--chat-ids", default="", help="Comma-separated chat IDs. Empty means rely on visible chats.")
    shortcut_lingo.add_argument("--include-visible-chats", action=argparse.BooleanOptionalAction, default=True)
    shortcut_lingo.add_argument("--recent-days", type=int, default=7, help="How many recent days of chat history to scan.")
    shortcut_lingo.add_argument("--force", action="store_true", help="Ignore last-run interval guard.")
    shortcut_lingo.add_argument("--judgements-file", default="", help="Optional AI review result JSON file. If provided, shortcut writes reviewed entries.")
    shortcut_lingo.add_argument("--publishable-only", action="store_true", help="Only sync create/append decisions when judgements are provided.")
    shortcut_lingo.add_argument("--source", default="lingo_shortcut")
    shortcut_lingo.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True, help="Write reviewed entries to Feishu Lingo remotely.")
    shortcut_lingo.add_argument("--write-local", action=argparse.BooleanOptionalAction, default=True, help="Mirror reviewed entries to local store.")
    shortcut_lingo.add_argument("--top-keywords", type=int, default=30)
    shortcut_lingo.add_argument("--candidate-limit", type=int, default=20)
    shortcut_lingo.set_defaults(func=cmd_shortcut_lingo_write)

    return parser


def cmd_collect_messages(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    require_user_auth(
        args,
        required_scopes=[
            "im:chat:read",
            "im:message:readonly",
            "im:message.group_msg:get_as_user",
        ],
    )
    result = collect_messages(
        client=build_user_feishu_client(args, require_token=True),
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
        token_file=get_token_file_arg(args),
    )
    result = _attach_auth_profile_bootstrap(args, result)
    result["ok"] = True
    result["profile_bootstrap"]["edit_command"] = (
        "sofree-knowledge assistant set-profile "
        "--role <角色> --persona <形象> --businesses <业务1,业务2> --interests <兴趣1,兴趣2>"
    )
    return result


def cmd_exchange_code(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    result = exchange_code_for_token(args.code_or_url, token_file=get_token_file_arg(args))
    result = _attach_auth_profile_bootstrap(args, result)
    result["ok"] = True
    return result


def cmd_auth_status(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    result = auth_status(token_file=get_token_file_arg(args))
    result["ok"] = True
    return result


def cmd_auth_login(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    if str(args.device_code or "").strip():
        result = resume_device_login(
            str(args.device_code).strip(),
            interval=int(args.interval),
            expires_in=int(args.expires_in),
            token_file=get_token_file_arg(args),
        )
    elif args.no_wait:
        result = start_device_login(scope=args.scope)
    else:
        result = device_login(
            scope=args.scope,
            token_file=get_token_file_arg(args),
            open_browser=not bool(args.no_browser),
        )
    result = _attach_auth_profile_bootstrap(args, result)
    result["ok"] = True
    return result


def prepare_env(args: argparse.Namespace) -> None:
    env_file = resolve_env_file(args.env_file, output_dir=args.output_dir)
    load_env_file(env_file)
    explicit_scope = str(getattr(args, "user_open_id", "") or "").strip()
    inferred_scope = explicit_scope or _infer_user_scope_from_token(args)
    if inferred_scope:
        args.user_open_id = inferred_scope
    scoped_output_dir = _resolve_user_scoped_output_dir(
        output_dir=str(getattr(args, "output_dir", ".") or "."),
        user_open_id=inferred_scope,
    )
    args.output_dir = str(scoped_output_dir)
    if not str(getattr(args, "token_file", "") or "").strip():
        scoped_token_file = _resolve_user_scoped_token_file(
            output_dir=str(scoped_output_dir),
            user_open_id=inferred_scope,
        )
        if scoped_token_file is not None:
            args.token_file = str(scoped_token_file)


def _sanitize_user_scope(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in normalized)


def _resolve_user_scoped_output_dir(*, output_dir: str, user_open_id: str) -> Path:
    root = Path(output_dir).expanduser()
    scope = _sanitize_user_scope(user_open_id)
    if not scope:
        return root
    return root / "users" / scope


def _resolve_user_scoped_token_file(*, output_dir: str, user_open_id: str) -> Path | None:
    scope = _sanitize_user_scope(user_open_id)
    if not scope:
        return None
    return Path(output_dir).expanduser() / "token.json"


def _infer_user_scope_from_token(args: argparse.Namespace) -> str:
    explicit_token_file = str(getattr(args, "token_file", "") or "").strip()
    identity = get_user_identity(token_file=explicit_token_file or None)
    return str(identity.get("open_id") or "").strip()


def get_token_file_arg(args: argparse.Namespace) -> str | None:
    value = str(getattr(args, "token_file", "") or "").strip()
    return value or None


def require_user_auth(args: argparse.Namespace, required_scopes: str | list[str] | tuple[str, ...]) -> dict[str, Any]:
    return ensure_user_auth(required_scopes=required_scopes, token_file=get_token_file_arg(args))


def _instantiate_feishu_client(
    *,
    user_access_token: str | None = None,
    token_file: str | Path | None = None,
) -> Any:
    try:
        if user_access_token is None:
            return FeishuClient(token_file=token_file)
        return FeishuClient(user_access_token=user_access_token, token_file=token_file)
    except TypeError:
        # Tests sometimes monkeypatch FeishuClient with a zero-arg callable.
        return FeishuClient()


def build_user_feishu_client(args: argparse.Namespace, *, require_token: bool = False) -> Any:
    token_file = get_token_file_arg(args)
    token = get_user_access_token(token_file=token_file)
    if require_token and not token:
        raise ValueError("Missing Feishu user access token. Run `sofree-knowledge auth login` first.")
    return _instantiate_feishu_client(user_access_token=token, token_file=token_file)


def build_bot_feishu_client() -> Any:
    return FeishuClient()


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def _resolve_push_target(
    args: argparse.Namespace,
    resolved_target_user_id: str,
) -> tuple[str, str]:
    return resolve_push_target(
        args,
        resolved_target_user_id=resolved_target_user_id,
        get_user_identity=get_user_identity,
    )


def _load_assistant_profile_config(args: argparse.Namespace) -> dict[str, Any]:
    return load_assistant_profile_config(output_dir=args.output_dir, profile_file=args.profile_file)


def _build_profile_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return build_profile_overrides(
        persona=args.persona,
        role=args.role,
        businesses=args.businesses,
        interests=args.interests,
    )


def _build_schedule_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return build_schedule_overrides(
        mode=args.mode,
        timezone=args.timezone,
        weekly_brief_cron=args.weekly_brief_cron,
        nightly_interest_cron=args.nightly_interest_cron,
        weekly_enabled=args.weekly_enabled,
        nightly_enabled=args.nightly_enabled,
    )


def cmd_assistant_set_profile(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    payload = {
        "profile": {
            **_build_profile_overrides(args),
            "require_user_confirmation": False,
        },
        "schedule": _build_schedule_overrides(args),
        "retrieval": build_retrieval_overrides(
            dual_tower_enabled=args.dual_tower_enabled,
            dual_tower_model=args.dual_tower_model,
            dual_tower_model_file=args.dual_tower_model_file,
            dual_tower_top_k=args.dual_tower_top_k,
            dual_tower_min_score=args.dual_tower_min_score,
        ),
    }
    path = save_assistant_profile_config(output_dir=args.output_dir, profile_file=args.profile_file, payload=payload)
    clean_result = {
        "ok": True,
        "profile_file": str(path),
        "profile": payload["profile"],
        "schedule": payload["schedule"],
        "retrieval": payload["retrieval"],
        "questionnaire_hint": [
            "请确认当前关注的业务方向（可多选）",
            "请确认最近更关注的话题关键词",
            "是否同意用最近阅读文档和群聊内容来更新画像",
        ],
    }
    return clean_result
    return {
        "ok": True,
        "profile_file": str(path),
        "profile": payload["profile"],
        "schedule": payload["schedule"],
        "retrieval": payload["retrieval"],
        "questionnaire_hint": [
            "请确认当前并行业务（可多选）",
            "请确认最近更关注的话题关键词",
            "是否同意用最近阅读文档建议更新画像",
        ],
    }


def _attach_auth_profile_bootstrap(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    token_payload = result.get("token")
    if not isinstance(token_payload, dict) or not token_payload.get("has_access_token"):
        return result
    token_file = get_token_file_arg(args)
    inferred_open_id = str(token_payload.get("open_id") or "").strip() or str(get_user_identity(token_file=token_file).get("open_id") or "").strip()
    if inferred_open_id:
        args.user_open_id = inferred_open_id
        args.output_dir = str(
            _resolve_user_scoped_output_dir(
                output_dir=str(getattr(args, "output_dir", ".") or "."),
                user_open_id=inferred_open_id,
            )
        )
        if not str(getattr(args, "token_file", "") or "").strip():
            scoped_token_file = _resolve_user_scoped_token_file(
                output_dir=str(args.output_dir),
                user_open_id=inferred_open_id,
            )
            if scoped_token_file is not None:
                args.token_file = str(scoped_token_file)
                token_file = str(scoped_token_file)
    client = FeishuClient.from_user_context(token_file=token_file, require_user_token=True)
    identity = get_user_identity(token_file=token_file)
    display_name = _lookup_user_display_name(client, identity)
    existing_config = load_assistant_profile_config(output_dir=args.output_dir, profile_file="")
    existing_profile = dict(existing_config.get("profile", {})) if isinstance(existing_config, dict) else {}
    try:
        online_inputs = collect_online_personal_inputs(
            client=client,
            target_user_id="",
            token_file=token_file or "",
            include_visible_chats=True,
            max_chats=8,
            max_messages_per_chat=30,
            max_drive_docs=10,
            max_knowledge=12,
            recent_days=14,
        )
    except Exception:
        online_inputs = {
            "documents": [],
            "messages": [],
            "knowledge_items": [],
            "meta": {"message_count": 0, "document_count": 0},
        }
    suggested_profile = suggest_profile_from_online_inputs(
        online_inputs=online_inputs,
        display_name=display_name,
        existing_profile=existing_profile,
    )
    merged_config = dict(existing_config) if isinstance(existing_config, dict) else {}
    merged_config["profile"] = suggested_profile
    merged_config.setdefault("schedule", {})
    merged_config.setdefault("retrieval", {})
    profile_path = save_assistant_profile_config(output_dir=args.output_dir, payload=merged_config)
    source_meta = dict(online_inputs.get("meta", {})) if isinstance(online_inputs.get("meta", {}), dict) else {}
    source_meta.setdefault("document_count", len(online_inputs.get("documents", []) or []))
    source_meta.setdefault("message_count", len(online_inputs.get("messages", []) or []))
    result["profile_bootstrap"] = {
        "profile_file": str(profile_path),
        "pending_confirmation": True,
        "profile": suggested_profile,
        "card": build_profile_review_card(profile=suggested_profile, source_meta=source_meta),
        "confirm_command": "sofree-knowledge assistant confirm-profile",
        "edit_command": "sofree-knowledge assistant set-profile --role <角色> --persona <形象> --businesses <业务1,业务2> --interests <兴趣1,兴趣2>",
    }
    return result


def _lookup_user_display_name(client: FeishuClient, identity: dict[str, Any]) -> str:
    open_id = str(identity.get("open_id") or "").strip()
    if not open_id:
        return ""
    path = f"/open-apis/contact/v3/users/{open_id}"
    params = {"user_id_type": "open_id"}
    try:
        data = client.request("GET", path, params=params)
    except Exception:
        try:
            data = client.request("GET", path, params=params, access_token=client.get_tenant_access_token())
        except Exception:
            return ""
    user = data.get("data", {}).get("user", {}) if isinstance(data, dict) else {}
    return str(user.get("name") or user.get("en_name") or "").strip()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "profile_file": str(path),
        "profile": payload["profile"],
        "schedule": payload["schedule"],
        "questionnaire_hint": [
            "请确认当前并行业务（可多选）",
            "请确认最近更关注的话题关键词",
            "是否同意用最近阅读文档建议更新画像",
        ],
    }


def cmd_assistant_get_profile(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    path = Path(args.profile_file).expanduser() if str(args.profile_file or "").strip() else assistant_profile_default_path(args.output_dir)
    parsed = _load_assistant_profile_config(args)
    return {
        "ok": True,
        "profile_file": str(path),
        "exists": path.exists(),
        "profile": parsed.get("profile", {}) if isinstance(parsed, dict) else {},
        "schedule": parsed.get("schedule", {}) if isinstance(parsed, dict) else {},
        "retrieval": parsed.get("retrieval", {}) if isinstance(parsed, dict) else {},
    }


def cmd_assistant_confirm_profile(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    parsed = _load_assistant_profile_config(args)
    profile = dict(parsed.get("profile", {})) if isinstance(parsed, dict) else {}
    profile["require_user_confirmation"] = False
    payload = dict(parsed) if isinstance(parsed, dict) else {}
    payload["profile"] = profile
    path = save_assistant_profile_config(output_dir=args.output_dir, profile_file=args.profile_file, payload=payload)
    return {
        "ok": True,
        "profile_file": str(path),
        "profile": profile,
        "confirmed": True,
    }


def cmd_assistant_export_dual_tower_samples(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    profile_config = _load_assistant_profile_config(args)
    user_profile = dict(profile_config.get("profile", {})) if isinstance(profile_config, dict) else {}
    user_profile.update(_build_profile_overrides(args))

    loaded_documents = load_json_file(args.documents_file)
    if isinstance(loaded_documents, dict) and isinstance(loaded_documents.get("documents"), list):
        documents = [item for item in loaded_documents["documents"] if isinstance(item, dict)]
    elif isinstance(loaded_documents, list):
        documents = [item for item in loaded_documents if isinstance(item, dict)]
    else:
        raise ValueError("--documents-file must contain a JSON array or {'documents': [...]} object")

    loaded_access = load_json_file(args.access_records_file)
    if not isinstance(loaded_access, list):
        raise ValueError("--access-records-file must contain a JSON array")
    access_records = [item for item in loaded_access if isinstance(item, dict)]

    loaded_messages = load_json_file(args.messages_file)
    if isinstance(loaded_messages, dict) and isinstance(loaded_messages.get("messages"), list):
        messages = [item for item in loaded_messages["messages"] if isinstance(item, dict)]
    elif isinstance(loaded_messages, list):
        messages = [item for item in loaded_messages if isinstance(item, dict)]
    else:
        raise ValueError("--messages-file must contain a JSON array or {'messages': [...]} object")

    result = export_dual_tower_samples(
        documents=documents,
        access_records=access_records,
        messages=messages,
        user_profile=user_profile,
        target_user_id=str(args.target_user_id or "").strip(),
        output_file=str(args.output_file or "").strip(),
    )
    result["ok"] = True
    return result


def cmd_assistant_train_dual_tower(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    samples = load_dual_tower_samples(args.samples_file)
    result = train_dual_tower_baseline(
        samples=samples,
        output_file=str(args.output_file or "").strip(),
        min_token_weight=float(args.min_token_weight or 0.0),
    )
    result["ok"] = True
    result["samples_file"] = str(args.samples_file)
    return result


def cmd_lingo_extract_contexts(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    messages = load_json_file(args.messages_file)
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
    contexts = load_json_file(args.contexts_file)
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
    raw_judgements = load_text_file(args.judgements_file)
    judgements = parse_lingo_judgements(raw_judgements)
    if args.publishable_only:
        judgements = publishable_lingo_judgements(judgements)
    return {
        "ok": True,
        "judgements": judgements,
        "count": len(judgements),
    }


def cmd_lingo_upsert(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    aliases = [item.strip() for item in str(args.aliases or "").split(",") if item.strip()]
    context_ids = [item.strip() for item in str(args.context_ids or "").split(",") if item.strip()]
    normalized_keyword = str(args.keyword or "").strip()
    normalized_type = str(args.type or "").strip().lower()
    normalized_value = str(args.value or "").strip()
    store = LingoStore(args.output_dir)
    existing = store.get_entry(normalized_keyword)
    remote_deleted: dict[str, Any] | None = None
    remote_created: dict[str, Any] | None = None
    resolved_entity_id = str(args.entity_id or "").strip()
    skipped_remote_create = False
    skip_reason = ""
    if (
        args.remote
        and not args.force_remote_create
        and existing
        and str(existing.get("entity_id") or "").strip()
        and str(existing.get("type") or "").strip().lower() == normalized_type
        and str(existing.get("value") or "").strip() == normalized_value
    ):
        skipped_remote_create = True
        resolved_entity_id = str(existing.get("entity_id") or resolved_entity_id)
        skip_reason = "duplicate_guard: same keyword/type/value already has remote entity_id in local mirror"
    if args.remote and not skipped_remote_create:
        client = _instantiate_feishu_client()
        replace_entity_id = str(args.replace_entity_id or "").strip()
        if replace_entity_id:
            remote_deleted = client.delete_lingo_entity(replace_entity_id)
        if normalized_type in {"key", "black"} and normalized_value:
            remote_created = client.create_lingo_entity(
                key=normalized_keyword,
                description=normalized_value,
                aliases=aliases,
                provider="sofree-knowledge-cli",
                outer_id=normalized_keyword,
            )
            resolved_entity_id = str(remote_created.get("entity_id") or resolved_entity_id)
        else:
            skipped_remote_create = True
            skip_reason = "remote_create_skipped: only key/black with non-empty value are written remotely"

    local_entry: dict[str, Any] | None = None
    if args.write_local:
        local_entry = store.upsert_entry(
            keyword=normalized_keyword,
            entry_type=normalized_type,
            value=normalized_value,
            aliases=aliases,
            source=args.source,
            entity_id=resolved_entity_id,
            context_ids=context_ids,
        )

    result: dict[str, Any] = {
        "ok": True,
        "assume_success": True,
        "remote_enabled": bool(args.remote),
        "local_enabled": bool(args.write_local),
        "keyword": normalized_keyword,
        "entity_id": resolved_entity_id,
        "verify_after_create": False,
        "remote_create_skipped": bool(skipped_remote_create),
        "remote_create_skip_reason": skip_reason,
        "note": (
            "Create code=0 already means remote create accepted. "
            "Search/list may still be empty due to admin review or repository visibility."
        ),
    }
    if remote_deleted is not None:
        result["remote_deleted"] = remote_deleted
    if remote_created is not None:
        result["remote_created"] = remote_created
    if local_entry is not None:
        result["entry"] = local_entry
        result["lingo_store_file"] = str(LingoStore(args.output_dir).path)
    return result


def cmd_lingo_delete(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    result: dict[str, Any] = {"ok": True, "remote_enabled": bool(args.remote), "local_enabled": bool(args.delete_local)}
    if args.remote:
        entity_id = str(args.entity_id or "").strip()
        if not entity_id:
            raise ValueError("--entity-id is required when --remote is enabled")
        result["remote"] = _instantiate_feishu_client().delete_lingo_entity(entity_id)
    if args.delete_local and str(args.keyword or "").strip():
        store = LingoStore(args.output_dir)
        local = store.delete_entry(args.keyword)
        result["local"] = local
        result["lingo_store_file"] = str(store.path)
    return result


def cmd_lingo_list(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    store = LingoStore(args.output_dir)
    entries = store.list_entries()[: max(0, int(args.limit))]
    return {
        "ok": True,
        "entries": entries,
        "count": len(entries),
        "lingo_store_file": str(store.path),
    }


def _sync_lingo_judgements(
    *,
    output_dir: str,
    judgements: list[dict[str, Any]],
    publishable_only: bool,
    source: str,
    force_remote_create: bool,
    remote: bool,
    write_local: bool,
) -> dict[str, Any]:
    normalized_judgements = publishable_lingo_judgements(judgements) if publishable_only else judgements
    store = LingoStore(output_dir)
    client = _instantiate_feishu_client() if remote else None
    upserted: list[dict[str, Any]] = []
    for item in normalized_judgements:
        if not isinstance(item, dict):
            continue
        keyword = str(item.get("keyword") or "").strip()
        entry_type = str(item.get("type") or "nothing").strip().lower()
        value = str(item.get("value") or "").strip()
        aliases = [str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()]
        context_ids = [
            str(context_id).strip()
            for context_id in item.get("context_ids", [])
            if str(context_id).strip()
        ]
        entity_id = ""
        remote_created: dict[str, Any] | None = None
        remote_create_skipped = False
        remote_skip_reason = ""
        existing = store.get_entry(keyword)
        if (
            client is not None
            and not force_remote_create
            and existing
            and str(existing.get("entity_id") or "").strip()
            and str(existing.get("type") or "").strip().lower() == entry_type
            and str(existing.get("value") or "").strip() == value
        ):
            remote_create_skipped = True
            remote_skip_reason = "duplicate_guard: same keyword/type/value already has remote entity_id in local mirror"
            entity_id = str(existing.get("entity_id") or "")
        if client is not None:
            if remote_create_skipped:
                pass
            elif entry_type in {"key", "black"} and value:
                remote_created = client.create_lingo_entity(
                    key=keyword,
                    description=value,
                    aliases=aliases,
                    provider="sofree-knowledge-cli",
                    outer_id=keyword,
                )
                entity_id = str(remote_created.get("entity_id") or "")
            else:
                remote_create_skipped = True
                remote_skip_reason = "remote_create_skipped: only key/black with non-empty value are written remotely"

        local_entry: dict[str, Any] | None = None
        if write_local:
            local_entry = store.upsert_entry(
                keyword=keyword,
                entry_type=entry_type,
                value=value,
                aliases=aliases,
                source=source,
                entity_id=entity_id,
                context_ids=context_ids,
            )
        upserted.append(
            {
                "keyword": keyword,
                "type": entry_type,
                "value": value,
                "aliases": aliases,
                "entity_id": entity_id,
                "remote_created": remote_created,
                "remote_create_skipped": remote_create_skipped,
                "remote_create_skip_reason": remote_skip_reason,
                "assume_success": bool(remote_created is not None),
                "entry": local_entry,
            }
        )

    return {
        "assume_success": True,
        "remote_enabled": bool(remote),
        "local_enabled": bool(write_local),
        "verify_after_create": False,
        "note": (
            "Remote create success is accepted as final success by default. "
            "No post-create list/search verification is performed."
        ),
        "count": len(upserted),
        "entries": upserted,
        "lingo_store_file": str(store.path),
    }


def cmd_lingo_sync_from_file(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    raw = load_text_file(args.input_file)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw

    if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
        parsed = parsed["items"]

    if isinstance(parsed, list):
        if parsed and all(isinstance(item, dict) and "keyword" in item and "type" in item for item in parsed):
            judgements = parsed
        else:
            judgements = parse_lingo_judgements(parsed)
    else:
        judgements = parse_lingo_judgements(parsed)

    result = _sync_lingo_judgements(
        output_dir=args.output_dir,
        judgements=judgements,
        publishable_only=bool(args.publishable_only),
        source=args.source,
        force_remote_create=bool(args.force_remote_create),
        remote=bool(args.remote),
        write_local=bool(args.write_local),
    )
    result["ok"] = True
    return result


def cmd_lingo_auto_sync(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    pipeline_result = run_lingo_auto_pipeline(
        client=build_user_feishu_client(args, require_token=False),
        output_dir=args.output_dir,
        recent_days=args.recent_days,
        min_run_interval_days=args.min_run_interval_days,
        force=bool(args.force),
        chat_ids=args.chat_ids or None,
        include_visible_chats=bool(args.include_visible_chats),
        start_time=args.start_time,
        end_time=args.end_time,
        max_chats=args.max_chats,
        max_messages_per_chat=args.max_messages_per_chat,
        page_size=args.page_size,
        top_keywords=args.top_keywords,
        candidate_limit=args.candidate_limit,
        min_frequency=args.min_frequency,
        min_contexts=args.min_contexts,
        context_before=args.context_before,
        context_after=args.context_after,
        max_contexts=args.max_contexts,
        analyzer_enabled=bool(args.analyzer_enabled),
    )
    if pipeline_result.get("skipped"):
        pipeline_result["ok"] = True
        return pipeline_result

    if not str(args.judgements_file or "").strip():
        return {
            "ok": True,
            **pipeline_result,
            "sync": {
                "performed": False,
                "reason": "missing_judgements_file",
                "next_step": "Review the emitted ai_review_prompt.txt, let your AI reviewer produce JSON judgements, then rerun with --judgements-file.",
            },
        }

    raw = load_text_file(args.judgements_file)
    judgements = parse_lingo_ai_review_judgements(raw)
    sync_result = sync_ai_review_judgements(
        output_dir=args.output_dir,
        judgements=judgements,
        source=args.source,
        remote=bool(args.remote),
        write_local=bool(args.write_local),
        force_remote_create=bool(args.force_remote_create),
        client=_instantiate_feishu_client() if args.remote else None,
        publishable_only=bool(args.publishable_only),
    )
    return {
        "ok": True,
        **pipeline_result,
        "judgements": judgements,
        "judgement_count": len(judgements),
        "sync": {
            "performed": True,
            **sync_result,
        },
    }


def cmd_confused_detect_candidates(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    messages = load_json_file(args.messages_file)
    reactions: list[dict[str, Any]] = []
    if args.reactions_file:
        loaded = load_json_file(args.reactions_file)
        if isinstance(loaded, list):
            reactions = [item for item in loaded if isinstance(item, dict)]
    confused_reaction_keys = {
        key.strip() for key in str(args.confused_reaction_keys or "").split(",") if key.strip()
    }

    candidates = detect_confused_candidates(
        messages=messages,
        target_message_id=args.target_message_id,
        reactions=reactions,
        confused_reaction_keys=confused_reaction_keys or None,
        max_followup_gap=args.max_followup_gap,
        max_candidates=args.max_candidates,
    )
    return {
        "ok": True,
        "target_message_id": args.target_message_id,
        "candidates": candidates,
        "count": len(candidates),
    }


def cmd_confused_build_judge_prompt(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    candidate = load_json_file(args.candidate_file)
    if not isinstance(candidate, dict):
        raise ValueError("--candidate-file must contain a single JSON object")
    prompt = build_confused_judge_prompt(candidate)
    return {
        "ok": True,
        "prompt": prompt,
    }


def cmd_confused_parse_judgement(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    raw = load_text_file(args.judgement_file)
    judgement = parse_confused_judgement(raw)
    return {
        "ok": True,
        "judgement": judgement,
        "inline_insert_text": format_inline_explanation(judgement.get("micro_explain", "")),
    }


def cmd_assistant_build_personal_brief(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    return build_personal_brief_command_result(
        args,
        load_json_file=load_json_file,
        collect_online_personal_inputs=collect_online_personal_inputs,
        build_personal_brief=build_personal_brief,
        get_user_identity=get_user_identity,
        client_factory=lambda: build_user_feishu_client(args, require_token=False),
        push_client_factory=build_bot_feishu_client,
    )


def cmd_assistant_recommend(args: argparse.Namespace) -> dict[str, Any]:
    prepare_env(args)
    return recommend_command_result(
        args,
        collect_online_personal_inputs=collect_online_personal_inputs,
        build_personal_brief=build_personal_brief,
        get_user_identity=get_user_identity,
        client_factory=lambda: build_user_feishu_client(args, require_token=False),
        push_client_factory=build_bot_feishu_client,
    )


def cmd_shortcut_brief(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    payload.update(
        command="brief",
        target_user_id="",
        max_docs=10,
        max_related=5,
        max_interest_items=8,
        mode="",
        timezone="",
        weekly_brief_cron="",
        nightly_interest_cron="",
        weekly_enabled=None,
        nightly_enabled=None,
        dual_tower_model="",
        dual_tower_model_file="",
        dual_tower_top_k=None,
        dual_tower_min_score=None,
        dual_tower_min_samples=20,
        push=True,
    )
    shortcut_args = argparse.Namespace(**payload)
    return cmd_assistant_recommend(shortcut_args)


def cmd_shortcut_lingo_write(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    payload.update(
        command="lingo-write",
        lingo_command="auto-sync",
        min_run_interval_days=7,
        start_time="",
        end_time="",
        max_chats=200,
        max_messages_per_chat=500,
        page_size=50,
        min_frequency=2,
        min_contexts=1,
        context_before=1,
        context_after=1,
        max_contexts=80,
        analyzer_enabled=True,
        force_remote_create=False,
    )
    shortcut_args = argparse.Namespace(**payload)
    return cmd_lingo_auto_sync(shortcut_args)


if __name__ == "__main__":
    raise SystemExit(main())
