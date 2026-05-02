from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .profile import (
    build_profile_review_card,
    build_profile_overrides,
    build_retrieval_overrides,
    build_schedule_overrides,
    load_assistant_profile_config,
    save_assistant_profile_config,
    suggest_profile_from_online_inputs,
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
PUSH_DEDUPE_WINDOW_SECONDS = 6 * 3600


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


def _resolve_effective_profile(
    *,
    args: argparse.Namespace,
    profile_config: JsonDict,
    user_profile: JsonDict,
    schedule: JsonDict,
    retrieval: JsonDict,
    documents: list[JsonDict],
    messages: list[JsonDict],
    knowledge_items: list[JsonDict],
    online_meta: JsonDict,
) -> JsonDict:
    if not _should_prompt_profile_setup(user_profile):
        return user_profile
    suggested_profile = suggest_profile_from_online_inputs(
        online_inputs={
            "documents": documents,
            "messages": messages,
            "knowledge_items": knowledge_items,
            "meta": online_meta,
        },
        display_name=str(user_profile.get("display_name") or "").strip(),
        existing_profile=user_profile,
    )
    persisted = dict(profile_config) if isinstance(profile_config, dict) else {}
    persisted["profile"] = suggested_profile
    persisted["schedule"] = schedule
    persisted["retrieval"] = retrieval
    save_assistant_profile_config(
        output_dir=args.output_dir,
        profile_file=getattr(args, "profile_file", ""),
        payload=persisted,
    )
    return suggested_profile


def build_personal_brief_command_result(
    args: argparse.Namespace,
    *,
    load_json_file: Callable[[str], Any],
    collect_online_personal_inputs: CollectorFn,
    build_personal_brief: BuilderFn,
    get_user_identity: IdentityFn,
    client_factory: ClientFactory,
    push_client_factory: ClientFactory | None = None,
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

    user_profile = _resolve_effective_profile(
        args=args,
        profile_config=profile_config,
        user_profile=user_profile,
        schedule=schedule,
        retrieval=retrieval,
        documents=documents,
        messages=messages,
        knowledge_items=knowledge_items,
        online_meta=online_meta,
    )

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
    profile_setup_required = _should_prompt_profile_setup(user_profile)
    base_meta["profile_required_before_recommendation"] = bool(profile_setup_required)
    if args.push:
        receive_id_type, receive_id = resolve_push_target(
            args,
            resolved_target_user_id=resolved_target_user_id,
            get_user_identity=get_user_identity,
        )
        client = push_client_factory() if push_client_factory is not None else client_factory()
        push_summary_card = bool(args.push_summary_card)
        push_interest_card = bool(args.push_interest_card)
        if not push_summary_card and not push_interest_card:
            push_interest_card = True
        summary_push_result: JsonDict | None = None
        interest_push_result: JsonDict | None = None
        profile_push_result: JsonDict | None = None
        push_skips: list[dict[str, str]] = []
        push_errors: list[dict[str, str]] = []
        if profile_setup_required:
            try:
                profile_push_result, was_skipped = _send_message_with_dedupe(
                    client=client,
                    output_dir=args.output_dir,
                    receive_id=receive_id,
                    receive_id_type=receive_id_type,
                    card_type="profile_setup",
                    content=_build_profile_setup_card(user_profile, online_meta),
                )
                if was_skipped:
                    push_skips.append({"card": "profile_setup", "reason": "duplicate_content"})
            except Exception as exc:
                push_errors.append({"card": "profile_setup", "error": str(exc)})
        else:
            if push_summary_card:
                try:
                    summary_push_result, was_skipped = _send_message_with_dedupe(
                        client=client,
                        output_dir=args.output_dir,
                        receive_id=receive_id,
                        receive_id_type=receive_id_type,
                        card_type="summary",
                        content=report["card"],
                    )
                    if was_skipped:
                        push_skips.append({"card": "summary", "reason": "duplicate_content"})
                except Exception as exc:
                    push_errors.append({"card": "summary", "error": str(exc)})
            if push_interest_card:
                try:
                    interest_push_result, was_skipped = _send_message_with_dedupe(
                        client=client,
                        output_dir=args.output_dir,
                        receive_id=receive_id,
                        receive_id_type=receive_id_type,
                        card_type="interest",
                        content=report["interest_card"],
                    )
                    if was_skipped:
                        push_skips.append({"card": "interest", "reason": "duplicate_content"})
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
            "summary_enabled": bool(push_summary_card) and not profile_setup_required,
            "summary_message_id": (summary_push_result or {}).get("message_id", ""),
            "interest_enabled": bool(push_interest_card) and not profile_setup_required,
            "interest_message_id": (interest_push_result or {}).get("message_id", ""),
            "profile_setup_prompted": profile_push_result is not None,
            "profile_setup_message_id": (profile_push_result or {}).get("message_id", ""),
            "recommendation_deferred_until_profile_confirmed": bool(profile_setup_required),
            "skipped": push_skips,
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
        if profile_setup_required and args.push:
            return {
                "ok": True,
                "meta": base_meta,
                "profile_setup_card": _build_profile_setup_card(user_profile, online_meta),
                "profile": user_profile,
                "report_deferred": True,
            }
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
        if profile_setup_required and args.push:
            return {
                "ok": True,
                "meta": base_meta,
                "card": _build_profile_setup_card(user_profile, online_meta),
            }
        return {
            "ok": True,
            "meta": base_meta,
            "card": report["card"],
            "interest_card": report["interest_card"],
        }
    if profile_setup_required and args.push:
        return {
            "ok": True,
            "meta": base_meta,
            "profile_setup_card": _build_profile_setup_card(user_profile, online_meta),
            "profile": user_profile,
            "report_deferred": True,
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
    push_client_factory: ClientFactory | None = None,
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
    behavior_documents = [item for item in online.get("behavior_documents", []) if isinstance(item, dict)]
    access_records = [item for item in online.get("access_records", []) if isinstance(item, dict)]
    messages = [item for item in online.get("messages", []) if isinstance(item, dict)]
    knowledge_items = [item for item in online.get("knowledge_items", []) if isinstance(item, dict)]
    resolved_target_user_id = str(online.get("resolved_target_user_id") or resolved_target_user_id)
    online_meta = online.get("meta", {}) if isinstance(online.get("meta", {}), dict) else {}

    user_profile = _resolve_effective_profile(
        args=args,
        profile_config=profile_config,
        user_profile=user_profile,
        schedule=schedule,
        retrieval=retrieval,
        documents=documents,
        messages=messages,
        knowledge_items=knowledge_items,
        online_meta=online_meta,
    )

    sample_history_file = Path(args.output_dir).expanduser() / "assistant_dual_tower_samples.jsonl"
    model_file = Path(args.output_dir).expanduser() / "assistant_dual_tower_model.json"
    current_samples = build_weak_supervision_samples(
        documents=behavior_documents or documents,
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
    profile_setup_required = _should_prompt_profile_setup(user_profile)
    base_meta["profile_required_before_recommendation"] = bool(profile_setup_required)
    if args.push:
        receive_id_type, receive_id = resolve_push_target(
            args,
            resolved_target_user_id=resolved_target_user_id,
            get_user_identity=get_user_identity,
        )
        client = push_client_factory() if push_client_factory is not None else client_factory()
        push_summary_card = bool(args.push_summary_card)
        push_interest_card = bool(args.push_interest_card)
        if not push_summary_card and not push_interest_card:
            push_interest_card = True
        summary_push_result: JsonDict | None = None
        interest_push_result: JsonDict | None = None
        profile_push_result: JsonDict | None = None
        push_skips: list[dict[str, str]] = []
        push_errors: list[dict[str, str]] = []
        if profile_setup_required:
            try:
                profile_push_result, was_skipped = _send_message_with_dedupe(
                    client=client,
                    output_dir=args.output_dir,
                    receive_id=receive_id,
                    receive_id_type=receive_id_type,
                    card_type="profile_setup",
                    content=_build_profile_setup_card(user_profile, online_meta),
                )
                if was_skipped:
                    push_skips.append({"card": "profile_setup", "reason": "duplicate_content"})
            except Exception as exc:
                push_errors.append({"card": "profile_setup", "error": str(exc)})
        else:
            if push_summary_card:
                try:
                    summary_push_result, was_skipped = _send_message_with_dedupe(
                        client=client,
                        output_dir=args.output_dir,
                        receive_id=receive_id,
                        receive_id_type=receive_id_type,
                        card_type="summary",
                        content=report["card"],
                    )
                    if was_skipped:
                        push_skips.append({"card": "summary", "reason": "duplicate_content"})
                except Exception as exc:
                    push_errors.append({"card": "summary", "error": str(exc)})
            if push_interest_card:
                try:
                    interest_push_result, was_skipped = _send_message_with_dedupe(
                        client=client,
                        output_dir=args.output_dir,
                        receive_id=receive_id,
                        receive_id_type=receive_id_type,
                        card_type="interest",
                        content=report["interest_card"],
                    )
                    if was_skipped:
                        push_skips.append({"card": "interest", "reason": "duplicate_content"})
                except Exception as exc:
                    push_errors.append({"card": "interest", "error": str(exc)})
        base_meta["push"] = {
            "enabled": True,
            "receive_id_type": receive_id_type,
            "receive_id": receive_id,
            "chat_id": (summary_push_result or interest_push_result or {}).get("chat_id", ""),
            "summary_enabled": bool(push_summary_card) and not profile_setup_required,
            "summary_message_id": (summary_push_result or {}).get("message_id", ""),
            "interest_enabled": bool(push_interest_card) and not profile_setup_required,
            "interest_message_id": (interest_push_result or {}).get("message_id", ""),
            "profile_setup_prompted": profile_push_result is not None,
            "profile_setup_message_id": (profile_push_result or {}).get("message_id", ""),
            "recommendation_deferred_until_profile_confirmed": bool(profile_setup_required),
            "skipped": push_skips,
            "errors": push_errors,
        }
    else:
        base_meta["push"] = {"enabled": False}

    if args.output_format == "json":
        if profile_setup_required and args.push:
            return {
                "ok": True,
                "meta": base_meta,
                "profile_setup_card": _build_profile_setup_card(user_profile, online_meta),
                "profile": user_profile,
                "report_deferred": True,
            }
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
        if profile_setup_required and args.push:
            return {
                "ok": True,
                "meta": base_meta,
                "card": _build_profile_setup_card(user_profile, online_meta),
            }
        return {"ok": True, "meta": base_meta, "card": report["card"], "interest_card": report["interest_card"]}
    if profile_setup_required and args.push:
        return {
            "ok": True,
            "meta": base_meta,
            "profile_setup_card": _build_profile_setup_card(user_profile, online_meta),
            "profile": user_profile,
            "report_deferred": True,
        }
    return {
        "ok": True,
        "meta": base_meta,
        "report": {**{k: v for k, v in report.items() if k != "doc_markdown"}, "docs": report.get("documents", [])},
    }


def _should_prompt_profile_setup(user_profile: JsonDict) -> bool:
    if bool(user_profile.get("require_user_confirmation")):
        return True
    interests = user_profile.get("interests")
    businesses = user_profile.get("businesses")
    return not interests or not businesses


def _build_profile_setup_card(user_profile: JsonDict, online_meta: JsonDict) -> JsonDict:
    interests = user_profile.get("interests") or []
    businesses = user_profile.get("businesses") or []
    clean_profile_payload = {
        "role": str(user_profile.get("role") or "待确认角色"),
        "persona": str(user_profile.get("persona") or "待确认形象"),
        "businesses": [str(item) for item in businesses if str(item).strip()],
        "interests": [str(item) for item in interests if str(item).strip()],
    }
    clean_source_meta = {
        "message_count": int(online_meta.get("message_count", 0) or 0),
        "document_count": int(online_meta.get("document_count", 0) or 0),
    }
    clean_card = build_profile_review_card(profile=clean_profile_payload, source_meta=clean_source_meta)
    clean_card["header"]["subtitle"]["content"] = "推荐前请先确认画像"
    return clean_card
    profile_payload = {
        "role": str(user_profile.get("role") or "待确认角色"),
        "persona": str(user_profile.get("persona") or "待确认形象"),
        "businesses": [str(item) for item in businesses if str(item).strip()],
        "interests": [str(item) for item in interests if str(item).strip()],
    }
    source_meta = {
        "message_count": int(online_meta.get("message_count", 0) or 0),
        "document_count": int(online_meta.get("document_count", 0) or 0),
    }
    card = build_profile_review_card(profile=profile_payload, source_meta=source_meta)
    card["header"]["subtitle"]["content"] = "推荐卡片已推送，请确认是否设置画像"
    profile_payload["role"] = str(user_profile.get("role") or "待确认角色")
    profile_payload["persona"] = str(user_profile.get("persona") or "待确认形象")
    card = build_profile_review_card(profile=profile_payload, source_meta=source_meta)
    card["header"]["subtitle"]["content"] = "推荐前请先确认画像"
    return card


def _send_message_with_dedupe(
    *,
    client: Any,
    output_dir: str,
    receive_id: str,
    receive_id_type: str,
    card_type: str,
    content: JsonDict,
) -> tuple[JsonDict | None, bool]:
    state_file = Path(output_dir).expanduser() / "assistant_push_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = _load_push_state(state_file)
    cache_key = f"{receive_id_type}:{receive_id}:{card_type}"
    payload_text = json.dumps(content, ensure_ascii=False, sort_keys=True)
    payload_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    current_bucket = state.get(cache_key, {})
    last_hash = str(current_bucket.get("payload_hash") or "")
    last_sent_at = int(current_bucket.get("sent_at") or 0)
    if last_hash == payload_hash and last_sent_at > 0:
        return None, True
    result = client.send_message(
        receive_id=receive_id,
        receive_id_type=receive_id_type,
        msg_type="interactive",
        content=content,
    )
    state[cache_key] = {
        "payload_hash": payload_hash,
        "sent_at": 1,
        "message_id": str((result or {}).get("message_id") or ""),
    }
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, False


def _load_push_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {}
    try:
        data = json.loads(state_file.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
