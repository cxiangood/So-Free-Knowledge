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
from .config import get_user_identity
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
from .lingo_store import LingoStore
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

    lingo_upsert = lingo_subparsers.add_parser("upsert", help="Create/update one lingo entry (remote by default).")
    lingo_upsert.add_argument("--keyword", required=True)
    lingo_upsert.add_argument("--type", required=True, choices=["key", "black", "confused", "nothing"])
    lingo_upsert.add_argument("--value", default="")
    lingo_upsert.add_argument("--aliases", default="", help="Comma-separated aliases.")
    lingo_upsert.add_argument("--source", default="manual")
    lingo_upsert.add_argument("--entity-id", default="")
    lingo_upsert.add_argument("--replace-entity-id", default="", help="Delete this remote entity id first, then create new one.")
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
    lingo_sync.add_argument("--remote", action=argparse.BooleanOptionalAction, default=True, help="Write each entry to Feishu Lingo remotely.")
    lingo_sync.add_argument("--write-local", action=argparse.BooleanOptionalAction, default=True, help="Mirror synced entries to local store.")
    lingo_sync.set_defaults(func=cmd_lingo_sync_from_file)

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
    assistant_build.add_argument("--max-docs", type=int, default=10, help="Maximum number of ranked documents.")
    assistant_build.add_argument("--max-related", type=int, default=5, help="Maximum number of related messages/knowledge per document.")
    assistant_build.add_argument("--push", action="store_true", help="Push assistant result to Feishu.")
    assistant_build.add_argument("--receive-chat-id", default="", help="Explicit target chat_id for push. If set, push to group chat.")
    assistant_build.add_argument("--receive-open-id", default="", help="Explicit target open_id for push. Used when --receive-chat-id is empty.")
    assistant_build.add_argument("--output-format", choices=["all", "json", "doc", "card"], default="all")
    assistant_build.set_defaults(func=cmd_assistant_build_personal_brief)

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


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def resolve_push_target(
    args: argparse.Namespace,
    resolved_target_user_id: str,
) -> tuple[str, str]:
    explicit_chat_id = str(args.receive_chat_id or "").strip()
    if explicit_chat_id:
        return "chat_id", explicit_chat_id

    explicit_open_id = str(args.receive_open_id or "").strip()
    if explicit_open_id:
        return "open_id", explicit_open_id

    identity = get_user_identity(token_file=args.token_file or None)
    open_id = str(identity.get("open_id") or "").strip()
    if open_id:
        return "open_id", open_id

    if str(resolved_target_user_id or "").strip().startswith("ou_"):
        return "open_id", str(resolved_target_user_id).strip()

    raise ValueError("Push target unresolved. Provide --receive-open-id or --receive-chat-id, or init token with open_id.")


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
    remote_deleted: dict[str, Any] | None = None
    remote_created: dict[str, Any] | None = None
    resolved_entity_id = str(args.entity_id or "").strip()
    if args.remote:
        client = FeishuClient()
        replace_entity_id = str(args.replace_entity_id or "").strip()
        if replace_entity_id:
            remote_deleted = client.delete_lingo_entity(replace_entity_id)
        remote_created = client.create_lingo_entity(
            key=str(args.keyword),
            description=str(args.value),
            aliases=aliases,
            provider="sofree-knowledge-cli",
            outer_id=str(args.keyword),
        )
        resolved_entity_id = str(remote_created.get("entity_id") or resolved_entity_id)

    local_entry: dict[str, Any] | None = None
    if args.write_local:
        store = LingoStore(args.output_dir)
        local_entry = store.upsert_entry(
            keyword=args.keyword,
            entry_type=args.type,
            value=args.value,
            aliases=aliases,
            source=args.source,
            entity_id=resolved_entity_id,
            context_ids=context_ids,
        )

    result: dict[str, Any] = {
        "ok": True,
        "remote_enabled": bool(args.remote),
        "local_enabled": bool(args.write_local),
        "keyword": str(args.keyword),
        "entity_id": resolved_entity_id,
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
        result["remote"] = FeishuClient().delete_lingo_entity(entity_id)
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

    if args.publishable_only:
        judgements = publishable_lingo_judgements(judgements)

    store = LingoStore(args.output_dir)
    client = FeishuClient() if args.remote else None
    upserted: list[dict[str, Any]] = []
    for item in judgements:
        if not isinstance(item, dict):
            continue
        keyword = str(item.get("keyword") or "").strip()
        entry_type = str(item.get("type") or "nothing").strip().lower()
        value = str(item.get("value") or "").strip()
        context_ids = [
            str(context_id).strip()
            for context_id in item.get("context_ids", [])
            if str(context_id).strip()
        ]
        entity_id = ""
        remote_created: dict[str, Any] | None = None
        if client is not None:
            remote_created = client.create_lingo_entity(
                key=keyword,
                description=value,
                aliases=[],
                provider="sofree-knowledge-cli",
                outer_id=keyword,
            )
            entity_id = str(remote_created.get("entity_id") or "")

        local_entry: dict[str, Any] | None = None
        if args.write_local:
            local_entry = store.upsert_entry(
                keyword=keyword,
                entry_type=entry_type,
                value=value,
                source=args.source,
                entity_id=entity_id,
                context_ids=context_ids,
            )
        upserted.append(
            {
                "keyword": keyword,
                "type": entry_type,
                "value": value,
                "entity_id": entity_id,
                "remote_created": remote_created,
                "entry": local_entry,
            }
        )
    return {
        "ok": True,
        "remote_enabled": bool(args.remote),
        "local_enabled": bool(args.write_local),
        "count": len(upserted),
        "entries": upserted,
        "lingo_store_file": str(store.path),
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
    documents: list[dict[str, Any]] = []
    access_records: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    knowledge_items: list[dict[str, Any]] = []
    resolved_target_user_id = str(args.target_user_id or "").strip()
    online_meta: dict[str, Any] = {}

    if args.online:
        online = collect_online_personal_inputs(
            client=FeishuClient(),
            target_user_id=resolved_target_user_id,
            token_file=args.token_file,
            chat_ids=args.chat_ids or None,
            include_visible_chats=bool(args.include_visible_chats),
            max_chats=args.max_chats,
            max_messages_per_chat=args.max_messages_per_chat,
            max_drive_docs=args.max_drive_docs,
            max_knowledge=args.max_knowledge,
        )
        documents = [item for item in online.get("documents", []) if isinstance(item, dict)]
        access_records = [item for item in online.get("access_records", []) if isinstance(item, dict)]
        messages = [item for item in online.get("messages", []) if isinstance(item, dict)]
        knowledge_items = [item for item in online.get("knowledge_items", []) if isinstance(item, dict)]
        resolved_target_user_id = str(online.get("resolved_target_user_id") or resolved_target_user_id)
        online_meta = online.get("meta", {}) if isinstance(online.get("meta", {}), dict) else {}
    else:
        if not args.documents_file:
            raise ValueError("--documents-file is required in offline mode, or use --online")
        loaded_documents = load_json_file(args.documents_file)
        if isinstance(loaded_documents, dict) and isinstance(loaded_documents.get("documents"), list):
            documents = loaded_documents["documents"]
        elif isinstance(loaded_documents, list):
            documents = loaded_documents
        else:
            raise ValueError("--documents-file must contain a JSON array or {'documents': [...]} object")

        if args.access_records_file:
            loaded_access = load_json_file(args.access_records_file)
            if isinstance(loaded_access, list):
                access_records = [item for item in loaded_access if isinstance(item, dict)]

        if args.messages_file:
            loaded_messages = load_json_file(args.messages_file)
            if isinstance(loaded_messages, dict) and isinstance(loaded_messages.get("messages"), list):
                messages = [item for item in loaded_messages["messages"] if isinstance(item, dict)]
            elif isinstance(loaded_messages, list):
                messages = [item for item in loaded_messages if isinstance(item, dict)]

        if args.knowledge_file:
            loaded_knowledge = load_json_file(args.knowledge_file)
            if isinstance(loaded_knowledge, dict) and isinstance(loaded_knowledge.get("items"), list):
                knowledge_items = [item for item in loaded_knowledge["items"] if isinstance(item, dict)]
            elif isinstance(loaded_knowledge, list):
                knowledge_items = [item for item in loaded_knowledge if isinstance(item, dict)]

        if not resolved_target_user_id:
            identity = get_user_identity(token_file=args.token_file or None)
            resolved_target_user_id = str(identity.get("open_id") or identity.get("user_id") or "")

    report = build_personal_brief(
        documents=documents,
        access_records=access_records,
        messages=messages,
        target_user_id=resolved_target_user_id,
        knowledge_items=knowledge_items,
        max_docs=args.max_docs,
        max_related=args.max_related,
    )

    base_meta = {
        "mode": "online" if args.online else "offline",
        "resolved_target_user_id": resolved_target_user_id,
        "inputs": {
            "document_count": len(documents),
            "access_record_count": len(access_records),
            "message_count": len(messages),
            "knowledge_item_count": len(knowledge_items),
        },
    }
    if online_meta:
        base_meta["online"] = online_meta
    if args.push:
        receive_id_type, receive_id = resolve_push_target(args, resolved_target_user_id=resolved_target_user_id)
        push_result = FeishuClient().send_message(
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            msg_type="interactive",
            content=report["card"],
        )
        base_meta["push"] = {
            "enabled": True,
            "receive_id_type": receive_id_type,
            "receive_id": receive_id,
            "message_id": push_result.get("message_id", ""),
            "chat_id": push_result.get("chat_id", ""),
        }
    else:
        base_meta["push"] = {"enabled": False}

    if args.output_format == "json":
        return {
            "ok": True,
            "meta": base_meta,
            "report": {k: v for k, v in report.items() if k not in {"doc_markdown", "card"}},
        }
    if args.output_format == "doc":
        return {"ok": True, "meta": base_meta, "doc_markdown": report["doc_markdown"]}
    if args.output_format == "card":
        return {"ok": True, "meta": base_meta, "card": report["card"]}
    return {"ok": True, "meta": base_meta, "report": report}


if __name__ == "__main__":
    raise SystemExit(main())
