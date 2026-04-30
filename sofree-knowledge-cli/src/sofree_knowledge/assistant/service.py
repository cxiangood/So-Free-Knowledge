from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .profile import (
    build_profile_overrides,
    build_retrieval_overrides,
    build_schedule_overrides,
    load_assistant_profile_config,
)
from .training import (
    append_dual_tower_samples,
    load_dual_tower_samples,
    summarize_dual_tower_samples,
    train_dual_tower_baseline,
)
from .dual_tower_dataset import build_weak_supervision_samples


JsonDict = dict[str, Any]
ClientFactory = Callable[[], Any]
CollectorFn = Callable[..., JsonDict]
BuilderFn = Callable[..., JsonDict]
IdentityFn = Callable[..., JsonDict]


def resolve_push_target(
    args: argparse.Namespace,
    *,
    resolved_target_user_id: str,
    get_user_identity: IdentityFn,
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


def build_personal_brief_command_result(
    args: argparse.Namespace,
    *,
    load_json_file: Callable[[str], Any],
    collect_online_personal_inputs: CollectorFn,
    build_personal_brief: BuilderFn,
    get_user_identity: IdentityFn,
    client_factory: ClientFactory,
) -> JsonDict:
    documents: list[JsonDict] = []
    access_records: list[JsonDict] = []
    messages: list[JsonDict] = []
    knowledge_items: list[JsonDict] = []
    resolved_target_user_id = str(args.target_user_id or "").strip()
    online_meta: JsonDict = {}
    profile_config = load_assistant_profile_config(output_dir=args.output_dir, profile_file=args.profile_file)
    user_profile = dict(profile_config.get("profile", {})) if isinstance(profile_config, dict) else {}
    user_profile.update(
        build_profile_overrides(
            persona=args.persona,
            role=args.role,
            businesses=args.businesses,
            interests=args.interests,
        )
    )
    schedule = dict(profile_config.get("schedule", {})) if isinstance(profile_config, dict) else {}
    schedule.update(
        build_schedule_overrides(
            mode=args.mode,
            timezone=args.timezone,
            weekly_brief_cron=args.weekly_brief_cron,
            nightly_interest_cron=args.nightly_interest_cron,
            weekly_enabled=args.weekly_enabled,
            nightly_enabled=args.nightly_enabled,
        )
    )
    retrieval = dict(profile_config.get("retrieval", {})) if isinstance(profile_config, dict) else {}
    retrieval.update(
        build_retrieval_overrides(
            dual_tower_enabled=args.dual_tower_enabled,
            dual_tower_model=args.dual_tower_model,
            dual_tower_model_file=args.dual_tower_model_file,
            dual_tower_top_k=args.dual_tower_top_k,
            dual_tower_min_score=args.dual_tower_min_score,
        )
    )

    if args.online:
        online = collect_online_personal_inputs(
            client=client_factory(),
            target_user_id=resolved_target_user_id,
            token_file=args.token_file,
            chat_ids=args.chat_ids or None,
            include_visible_chats=bool(args.include_visible_chats),
            max_chats=args.max_chats,
            max_messages_per_chat=args.max_messages_per_chat,
            max_drive_docs=args.max_drive_docs,
            max_knowledge=args.max_knowledge,
            recent_days=args.recent_days,
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
        user_profile=user_profile,
        schedule=schedule,
        max_interest_items=args.max_interest_items,
        dual_tower_config=retrieval,
    )

    base_meta: JsonDict = {
        "mode": "online" if args.online else "offline",
        "recent_days": int(args.recent_days),
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
        receive_id_type, receive_id = resolve_push_target(
            args,
            resolved_target_user_id=resolved_target_user_id,
            get_user_identity=get_user_identity,
        )
        client = client_factory()
        push_summary_card = bool(args.push_summary_card)
        push_interest_card = bool(args.push_interest_card)
        if not push_summary_card and not push_interest_card:
            push_interest_card = True
        summary_push_result: JsonDict | None = None
        interest_push_result: JsonDict | None = None
        push_errors: list[dict[str, str]] = []
        if push_summary_card:
            try:
                summary_push_result = client.send_message(
                    receive_id=receive_id,
                    receive_id_type=receive_id_type,
                    msg_type="interactive",
                    content=report["card"],
                )
            except Exception as exc:
                push_errors.append({"card": "summary", "error": str(exc)})
        if push_interest_card:
            try:
                interest_push_result = client.send_message(
                    receive_id=receive_id,
                    receive_id_type=receive_id_type,
                    msg_type="interactive",
                    content=report["interest_card"],
                )
            except Exception as exc:
                push_errors.append({"card": "interest", "error": str(exc)})
        base_meta["push"] = {
            "enabled": True,
            "receive_id_type": receive_id_type,
            "receive_id": receive_id,
            "chat_id": (summary_push_result or interest_push_result or {}).get("chat_id", ""),
            "doc_push_enabled": False,
            "doc_push_skipped": True,
            "docs_status": "disabled",
            "docs": [],
            "summary_enabled": bool(push_summary_card),
            "summary_message_id": (summary_push_result or {}).get("message_id", ""),
            "interest_enabled": bool(push_interest_card),
            "interest_message_id": (interest_push_result or {}).get("message_id", ""),
            "errors": push_errors,
        }
    else:
        base_meta["push"] = {
            "enabled": False,
            "doc_push_enabled": False,
            "doc_push_skipped": True,
            "docs_status": "disabled",
            "docs": [],
        }

    if args.output_format == "json":
        filtered = {k: v for k, v in report.items() if k not in {"doc_markdown", "card", "interest_card"}}
        filtered["docs"] = filtered.get("documents", [])
        return {
            "ok": True,
            "meta": base_meta,
            "report": filtered,
        }
    if args.output_format == "doc":
        return {
            "ok": True,
            "meta": base_meta,
            "doc_markdown": report["doc_markdown"],
            "deprecated": {
                "output_format_doc": True,
                "message": "output-format=doc is deprecated; prefer output-format=card or output-format=all.",
            },
        }
    if args.output_format == "card":
        return {
            "ok": True,
            "meta": base_meta,
            "card": report["card"],
            "interest_card": report["interest_card"],
        }
    return {
        "ok": True,
        "meta": base_meta,
        "report": {
            **{k: v for k, v in report.items() if k != "doc_markdown"},
            "docs": report.get("documents", []),
        },
    }


def recommend_command_result(
    args: argparse.Namespace,
    *,
    collect_online_personal_inputs: CollectorFn,
    build_personal_brief: BuilderFn,
    get_user_identity: IdentityFn,
    client_factory: ClientFactory,
) -> JsonDict:
    profile_config = load_assistant_profile_config(output_dir=args.output_dir, profile_file=args.profile_file)
    user_profile = dict(profile_config.get("profile", {})) if isinstance(profile_config, dict) else {}
    user_profile.update(
        build_profile_overrides(
            persona=args.persona,
            role=args.role,
            businesses=args.businesses,
            interests=args.interests,
        )
    )
    schedule = dict(profile_config.get("schedule", {})) if isinstance(profile_config, dict) else {}
    schedule.update(
        build_schedule_overrides(
            mode=args.mode,
            timezone=args.timezone,
            weekly_brief_cron=args.weekly_brief_cron,
            nightly_interest_cron=args.nightly_interest_cron,
            weekly_enabled=args.weekly_enabled,
            nightly_enabled=args.nightly_enabled,
        )
    )
    retrieval = dict(profile_config.get("retrieval", {})) if isinstance(profile_config, dict) else {}
    retrieval.update(
        build_retrieval_overrides(
            dual_tower_enabled=None,
            dual_tower_model=args.dual_tower_model,
            dual_tower_model_file=args.dual_tower_model_file,
            dual_tower_top_k=args.dual_tower_top_k,
            dual_tower_min_score=args.dual_tower_min_score,
        )
    )

    resolved_target_user_id = str(args.target_user_id or "").strip()
    online = collect_online_personal_inputs(
        client=client_factory(),
        target_user_id=resolved_target_user_id,
        token_file=args.token_file,
        chat_ids=args.chat_ids or None,
        include_visible_chats=bool(args.include_visible_chats),
        max_chats=args.max_chats,
        max_messages_per_chat=args.max_messages_per_chat,
        max_drive_docs=args.max_drive_docs,
        max_knowledge=args.max_knowledge,
        recent_days=args.recent_days,
    )
    documents = [item for item in online.get("documents", []) if isinstance(item, dict)]
    access_records = [item for item in online.get("access_records", []) if isinstance(item, dict)]
    messages = [item for item in online.get("messages", []) if isinstance(item, dict)]
    knowledge_items = [item for item in online.get("knowledge_items", []) if isinstance(item, dict)]
    resolved_target_user_id = str(online.get("resolved_target_user_id") or resolved_target_user_id)
    online_meta = online.get("meta", {}) if isinstance(online.get("meta", {}), dict) else {}

    sample_history_file = Path(args.output_dir).expanduser() / "assistant_dual_tower_samples.jsonl"
    model_file = Path(args.output_dir).expanduser() / "assistant_dual_tower_model.json"
    current_samples = build_weak_supervision_samples(
        documents=documents,
        access_records=access_records,
        messages=messages,
        user_profile=user_profile,
        target_user_id=resolved_target_user_id,
    )
    append_result = append_dual_tower_samples(current_samples, str(sample_history_file)) if current_samples else {
        "output_file": str(sample_history_file),
        "appended_count": 0,
    }
    all_samples = load_dual_tower_samples(str(sample_history_file)) if sample_history_file.exists() else []
    min_samples = max(1, int(args.dual_tower_min_samples))
    enough_data = len(all_samples) >= min_samples
    auto_train_result: JsonDict = {
        "sample_history_file": str(sample_history_file),
        "model_file": str(model_file),
        "current_run_sample_count": len(current_samples),
        "accumulated_sample_count": len(all_samples),
        "min_samples_required": min_samples,
        "enough_data": enough_data,
        "append": append_result,
        "sample_summary": summarize_dual_tower_samples(all_samples),
    }
    if enough_data:
        train_result = train_dual_tower_baseline(samples=all_samples, output_file=str(model_file))
        auto_train_result["train"] = train_result
        retrieval["enabled"] = True
        retrieval["model_file"] = str(model_file)
    else:
        retrieval["enabled"] = False
        retrieval["model_file"] = ""

    report = build_personal_brief(
        documents=documents,
        access_records=access_records,
        messages=messages,
        target_user_id=resolved_target_user_id,
        knowledge_items=knowledge_items,
        max_docs=args.max_docs,
        max_related=args.max_related,
        user_profile=user_profile,
        schedule=schedule,
        max_interest_items=args.max_interest_items,
        dual_tower_config=retrieval,
    )

    base_meta: JsonDict = {
        "mode": "online",
        "recent_days": int(args.recent_days),
        "resolved_target_user_id": resolved_target_user_id,
        "inputs": {
            "document_count": len(documents),
            "access_record_count": len(access_records),
            "message_count": len(messages),
            "knowledge_item_count": len(knowledge_items),
        },
        "online": online_meta,
        "auto_retrieval": auto_train_result,
    }
    if args.push:
        receive_id_type, receive_id = resolve_push_target(
            args,
            resolved_target_user_id=resolved_target_user_id,
            get_user_identity=get_user_identity,
        )
        client = client_factory()
        push_summary_card = bool(args.push_summary_card)
        push_interest_card = bool(args.push_interest_card)
        if not push_summary_card and not push_interest_card:
            push_interest_card = True
        summary_push_result: JsonDict | None = None
        interest_push_result: JsonDict | None = None
        push_errors: list[dict[str, str]] = []
        if push_summary_card:
            try:
                summary_push_result = client.send_message(
                    receive_id=receive_id,
                    receive_id_type=receive_id_type,
                    msg_type="interactive",
                    content=report["card"],
                )
            except Exception as exc:
                push_errors.append({"card": "summary", "error": str(exc)})
        if push_interest_card:
            try:
                interest_push_result = client.send_message(
                    receive_id=receive_id,
                    receive_id_type=receive_id_type,
                    msg_type="interactive",
                    content=report["interest_card"],
                )
            except Exception as exc:
                push_errors.append({"card": "interest", "error": str(exc)})
        base_meta["push"] = {
            "enabled": True,
            "receive_id_type": receive_id_type,
            "receive_id": receive_id,
            "chat_id": (summary_push_result or interest_push_result or {}).get("chat_id", ""),
            "summary_enabled": bool(push_summary_card),
            "summary_message_id": (summary_push_result or {}).get("message_id", ""),
            "interest_enabled": bool(push_interest_card),
            "interest_message_id": (interest_push_result or {}).get("message_id", ""),
            "errors": push_errors,
        }
    else:
        base_meta["push"] = {"enabled": False}

    if args.output_format == "json":
        filtered = {k: v for k, v in report.items() if k not in {"doc_markdown", "card", "interest_card"}}
        filtered["docs"] = filtered.get("documents", [])
        return {"ok": True, "meta": base_meta, "report": filtered}
    if args.output_format == "doc":
        return {
            "ok": True,
            "meta": base_meta,
            "doc_markdown": report["doc_markdown"],
            "deprecated": {
                "output_format_doc": True,
                "message": "output-format=doc is deprecated; prefer output-format=card or output-format=all.",
            },
        }
    if args.output_format == "card":
        return {"ok": True, "meta": base_meta, "card": report["card"], "interest_card": report["interest_card"]}
    return {
        "ok": True,
        "meta": base_meta,
        "report": {**{k: v for k, v in report.items() if k != "doc_markdown"}, "docs": report.get("documents", [])},
    }
