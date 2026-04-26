"""Personal assistant brief builder from docs/access/messages/knowledge."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


URGENCY_KEYWORDS = (
    "紧急",
    "马上",
    "尽快",
    "asap",
    "今天",
    "今晚",
    "截止",
    "ddl",
    "线上",
    "故障",
    "报错",
    "异常",
    "阻塞",
    "p0",
    "p1",
)

TODO_KEYWORDS = (
    "todo",
    "to do",
    "待办",
    "待跟进",
    "跟进",
    "action item",
    "行动项",
    "需要",
    "请在",
    "请于",
)

DOC_TYPE_PATTERNS = (
    ("minutes", ("/minutes/", "会议纪要", "纪要", "minutes")),
    ("sheet", ("/sheet/", "/sheets/", "表格", "sheet", "excel")),
    ("base", ("/base/", "多维表格", "bitable", "base")),
    ("wiki", ("/wiki/", "知识库", "wiki")),
    ("doc", ("/docx/", "/docs/", "文档", "doc")),
)

INTEREST_SIGNAL_KEYWORDS = (
    "需求",
    "上线",
    "风险",
    "客户",
    "review",
    "排期",
    "交付",
    "发布",
    "故障",
    "截止",
    "待办",
)

INTEREST_NOISE_PATTERNS = (
    "llm 调用失败",
    "too many requests",
    "429",
    "[plan_command]",
    "write_policy:",
    "output_format:",
    "llm 配置不完整",
    "缺少:",
    "我没办法直接帮你",
    "可以和我说说",
    "我是有什么问题",
    '"json_card"',
    "{\\\"json_card\\\"",
)


def build_personal_brief(
    documents: list[dict[str, Any]],
    access_records: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    knowledge_items: list[dict[str, Any]] | None = None,
    target_user_id: str = "",
    max_docs: int = 10,
    max_related: int = 5,
    user_profile: dict[str, Any] | None = None,
    schedule: dict[str, Any] | None = None,
    max_interest_items: int = 8,
) -> dict[str, Any]:
    docs = [_normalize_doc(item) for item in documents]
    records = [item for item in (access_records or []) if isinstance(item, dict)]
    msgs = [_normalize_message(item) for item in (messages or []) if isinstance(item, dict)]
    knowledge = [_normalize_knowledge(item) for item in (knowledge_items or []) if isinstance(item, dict)]
    profile = _normalize_profile(user_profile or {})
    normalized_schedule = _normalize_schedule(schedule or {})

    doc_stats = _build_doc_stats(records, target_user_id=target_user_id)

    ranked_docs: list[dict[str, Any]] = []
    for doc in docs:
        doc_type = _infer_doc_type(doc)
        title_keywords = _extract_keywords(" ".join([doc["title"], doc["summary"]]))
        related_messages = _rank_related_messages(doc, title_keywords, msgs, max_related=max_related)
        related_knowledge = _rank_related_knowledge(title_keywords, knowledge, max_related=max_related)

        urgency_score = _score_doc_urgency(doc=doc, related_messages=related_messages)
        recommend_score = _score_doc_recommend(
            doc=doc,
            stats=doc_stats.get(doc["doc_id"], {}),
            related_messages=related_messages,
            related_knowledge=related_knowledge,
        )
        business = _infer_business(doc, profile["business_tracks"])
        todo_items = _extract_todo_items(doc, related_messages)
        ranked_docs.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "url": doc["url"],
                "summary": doc["summary"],
                "doc_type": doc_type,
                "business": business,
                "recommend_score": recommend_score,
                "urgency_score": urgency_score,
                "urgency_stars": _urgency_stars(urgency_score),
                "priority": _priority_label(urgency_score, recommend_score),
                "reasons": _build_reasons(doc, doc_stats.get(doc["doc_id"], {}), related_messages),
                "todo_items": todo_items,
                "related_messages": related_messages,
                "related_knowledge": related_knowledge,
            }
        )

    ranked_docs.sort(
        key=lambda item: (int(item["urgency_score"]), int(item["recommend_score"])),
        reverse=True,
    )
    ranked_docs = ranked_docs[:max_docs]

    grouped_documents = _group_documents_by_business(ranked_docs)
    interest_digest = _build_interest_digest(msgs, profile["interests"], max_items=max_interest_items)

    summary = {
        "doc_count": len(ranked_docs),
        "critical_count": sum(1 for item in ranked_docs if item["priority"] == "critical"),
        "high_count": sum(1 for item in ranked_docs if item["priority"] == "high"),
        "normal_count": sum(1 for item in ranked_docs if item["priority"] == "normal"),
        "todo_count": sum(len(item.get("todo_items", [])) for item in ranked_docs),
        "business_count": len(grouped_documents),
        "interest_hit_count": len(interest_digest.get("items", [])),
    }

    doc_markdown = _to_markdown(ranked_docs, grouped_documents=grouped_documents, profile=profile)
    card = _to_card(ranked_docs, profile=profile)
    interest_card = _to_interest_card(interest_digest, profile=profile)
    runtime_plan = _build_runtime_plan(normalized_schedule)
    return {
        "summary": summary,
        "documents": ranked_docs,
        "grouped_documents": grouped_documents,
        "profile": profile,
        "schedule": normalized_schedule,
        "runtime_plan": runtime_plan,
        "interest_digest": interest_digest,
        "doc_markdown": doc_markdown,
        "card": card,
        "interest_card": interest_card,
    }


def _normalize_doc(item: dict[str, Any]) -> dict[str, str]:
    return {
        "doc_id": str(item.get("doc_id") or item.get("id") or item.get("token") or ""),
        "title": str(item.get("title") or item.get("name") or ""),
        "summary": str(item.get("summary") or item.get("description") or ""),
        "url": str(item.get("url") or item.get("link") or ""),
        "updated_at": str(item.get("updated_at") or item.get("update_time") or item.get("timestamp") or ""),
    }


def _normalize_message(item: dict[str, Any]) -> dict[str, str]:
    text = item.get("text")
    if text is None:
        text = item.get("content", "")
    summary_text = (
        item.get("openclaw_summary")
        or item.get("paraphrase")
        or item.get("summary_text")
        or item.get("summary")
        or ""
    )
    return {
        "message_id": str(item.get("message_id") or item.get("id") or ""),
        "chat_id": str(item.get("chat_id") or ""),
        "text": str(text or ""),
        "summary_text": str(summary_text or ""),
        "create_time": str(item.get("create_time") or item.get("timestamp") or ""),
    }


def _normalize_knowledge(item: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(item.get("id") or item.get("knowledge_id") or ""),
        "title": str(item.get("title") or item.get("keyword") or ""),
        "content": str(item.get("content") or item.get("value") or item.get("summary") or ""),
    }


def _normalize_profile(item: dict[str, Any]) -> dict[str, Any]:
    businesses_raw = item.get("businesses") or item.get("business_tracks") or []
    business_tracks: list[dict[str, Any]] = []
    if isinstance(businesses_raw, list):
        for idx, entry in enumerate(businesses_raw, start=1):
            if isinstance(entry, str):
                name = entry.strip()
                if not name:
                    continue
                business_tracks.append({"id": f"biz_{idx}", "name": name, "keywords": list(_extract_keywords(name))})
                continue
            if isinstance(entry, dict):
                name = str(entry.get("name") or entry.get("business") or "").strip()
                if not name:
                    continue
                keywords = entry.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [x.strip() for x in keywords.split(",") if x.strip()]
                if not isinstance(keywords, list):
                    keywords = []
                seed = list(_extract_keywords(" ".join([name, " ".join(str(k) for k in keywords)])))
                business_tracks.append(
                    {
                        "id": str(entry.get("id") or f"biz_{idx}"),
                        "name": name,
                        "keywords": seed,
                    }
                )

    interests_raw = item.get("interests") or []
    interests: list[str] = []
    if isinstance(interests_raw, str):
        interests = [value.strip() for value in interests_raw.split(",") if value.strip()]
    elif isinstance(interests_raw, list):
        interests = [str(value).strip() for value in interests_raw if str(value).strip()]

    return {
        "persona": str(item.get("persona") or item.get("avatar") or "").strip(),
        "role": str(item.get("role") or item.get("profession") or "").strip(),
        "business_tracks": business_tracks,
        "interests": interests,
        "current_focus": [track["name"] for track in business_tracks],
        "require_user_confirmation": bool(item.get("require_user_confirmation", True)),
        "update_hint": "根据最新阅读文档候选更新画像时，应先向用户确认。",
    }


def _normalize_schedule(item: dict[str, Any]) -> dict[str, Any]:
    mode = str(item.get("mode") or "scheduled").strip().lower()
    if mode not in {"scheduled", "manual", "hybrid"}:
        mode = "scheduled"
    timezone_name = str(item.get("timezone") or "Asia/Shanghai").strip() or "Asia/Shanghai"
    weekly_enabled = bool(item.get("weekly_enabled", mode in {"scheduled", "hybrid"}))
    nightly_enabled = bool(item.get("nightly_enabled", mode in {"scheduled", "hybrid"}))
    return {
        "mode": mode,
        "timezone": timezone_name,
        "weekly_brief_cron": str(item.get("weekly_brief_cron") or "0 9 * * MON"),
        "nightly_interest_cron": str(item.get("nightly_interest_cron") or "0 21 * * *"),
        "weekly_enabled": weekly_enabled,
        "nightly_enabled": nightly_enabled,
        "manual_supported": True,
    }


def _build_doc_stats(records: list[dict[str, Any]], target_user_id: str) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"views": 0, "edits": 0, "comments": 0})
    normalized_target = str(target_user_id or "").strip()
    for record in records:
        doc_id = str(record.get("doc_id") or record.get("document_id") or record.get("token") or "").strip()
        if not doc_id:
            continue
        if normalized_target:
            user_id = str(record.get("user_id") or record.get("operator_id") or "").strip()
            if user_id and user_id != normalized_target:
                continue
        action = str(record.get("action") or record.get("event") or "").lower()
        count = int(record.get("count") or 1)
        if "edit" in action:
            stats[doc_id]["edits"] += count
        elif "comment" in action:
            stats[doc_id]["comments"] += count
        else:
            stats[doc_id]["views"] += count
    return stats


def _extract_keywords(text: str) -> set[str]:
    normalized = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    result: set[str] = {token for token in tokens if token.strip()}
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            for idx in range(0, len(token) - 1):
                result.add(token[idx : idx + 2])
    return result


def _rank_related_messages(
    doc: dict[str, str],
    doc_keywords: set[str],
    messages: list[dict[str, str]],
    max_related: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        text = message["text"]
        if not text:
            continue
        message_keywords = _extract_keywords(text)
        overlap = len(doc_keywords & message_keywords)
        if overlap <= 0:
            continue
        urgency_hit = any(keyword in text.lower() for keyword in URGENCY_KEYWORDS)
        score = overlap * 20 + (30 if urgency_hit else 0)
        items.append(
            {
                "message_id": message["message_id"],
                "chat_id": message["chat_id"],
                "summary": _trim(text, 100),
                "score": score,
                "create_time": message["create_time"],
            }
        )
    items.sort(key=lambda item: (int(item["score"]), str(item["create_time"])), reverse=True)
    return items[:max_related]


def _rank_related_knowledge(
    doc_keywords: set[str],
    knowledge: list[dict[str, str]],
    max_related: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in knowledge:
        text = " ".join([entry["title"], entry["content"]])
        keywords = _extract_keywords(text)
        overlap = len(doc_keywords & keywords)
        if overlap <= 0:
            continue
        score = overlap * 25
        items.append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "summary": _trim(entry["content"], 80),
                "score": score,
            }
        )
    items.sort(key=lambda item: int(item["score"]), reverse=True)
    return items[:max_related]


def _score_doc_urgency(doc: dict[str, str], related_messages: list[dict[str, Any]]) -> int:
    score = 0
    text = f"{doc['title']} {doc['summary']}".lower()
    if any(keyword in text for keyword in URGENCY_KEYWORDS):
        score += 40
    score += min(40, sum(1 for item in related_messages if int(item.get("score", 0)) >= 40) * 10)
    score += min(
        20,
        sum(
            1
            for item in related_messages
            if any(keyword in str(item.get("summary", "")).lower() for keyword in URGENCY_KEYWORDS)
        )
        * 10,
    )
    return min(100, score)


def _score_doc_recommend(
    doc: dict[str, str],
    stats: dict[str, int],
    related_messages: list[dict[str, Any]],
    related_knowledge: list[dict[str, Any]],
) -> int:
    score = 20
    score += min(25, int(stats.get("views", 0)) * 3)
    score += min(20, int(stats.get("edits", 0)) * 8)
    score += min(15, int(stats.get("comments", 0)) * 5)
    score += min(10, len(related_messages) * 2)
    score += min(10, len(related_knowledge) * 3)
    if _is_recent(doc.get("updated_at", "")):
        score += 10
    return min(100, score)


def _priority_label(urgency_score: int, recommend_score: int) -> str:
    if urgency_score >= 70:
        return "critical"
    if urgency_score >= 45 or recommend_score >= 75:
        return "high"
    return "normal"


def _urgency_stars(urgency_score: int) -> int:
    if urgency_score >= 85:
        return 5
    if urgency_score >= 70:
        return 4
    if urgency_score >= 50:
        return 3
    if urgency_score >= 30:
        return 2
    return 1


def _build_reasons(doc: dict[str, str], stats: dict[str, int], related_messages: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if int(stats.get("edits", 0)) > 0:
        reasons.append(f"近期编辑活跃（{stats.get('edits', 0)} 次）")
    if int(stats.get("views", 0)) > 0:
        reasons.append(f"近期访问较多（{stats.get('views', 0)} 次）")
    if related_messages:
        reasons.append(f"关联群聊消息 {len(related_messages)} 条")
    if not reasons:
        reasons.append("基础推荐：与近期语义上下文有关联")
    return reasons


def _infer_doc_type(doc: dict[str, str]) -> str:
    text = " ".join([doc.get("url", ""), doc.get("title", ""), doc.get("summary", "")]).lower()
    for doc_type, patterns in DOC_TYPE_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return doc_type
    return "file"


def _infer_business(doc: dict[str, str], business_tracks: list[dict[str, Any]]) -> str:
    if not business_tracks:
        return "General"
    doc_keywords = _extract_keywords(" ".join([doc.get("title", ""), doc.get("summary", ""), doc.get("url", "")]))
    best_name = "General"
    best_score = 0
    for track in business_tracks:
        track_keywords = {str(item).lower() for item in track.get("keywords", []) if str(item).strip()}
        overlap = len(doc_keywords & track_keywords)
        if overlap > best_score:
            best_score = overlap
            best_name = str(track.get("name") or "General")
    return best_name


def _extract_todo_items(doc: dict[str, Any], related_messages: list[dict[str, Any]]) -> list[str]:
    source_texts = [str(doc.get("summary", ""))]
    source_texts.extend(str(item.get("summary", "")) for item in related_messages)
    todos: list[str] = []
    for text in source_texts:
        if not text:
            continue
        lowered = text.lower()
        if any(keyword in lowered for keyword in TODO_KEYWORDS):
            todos.append(_trim(text, 80))
    deduped: list[str] = []
    seen: set[str] = set()
    for todo in todos:
        if todo not in seen:
            seen.add(todo)
            deduped.append(todo)
    return deduped[:3]


def _group_documents_by_business(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in documents:
        business = str(item.get("business") or "General")
        buckets[business].append(item)
    grouped: list[dict[str, Any]] = []
    for business, docs in buckets.items():
        docs.sort(
            key=lambda item: (
                int(item.get("urgency_stars", 1)),
                int(item.get("urgency_score", 0)),
                int(item.get("recommend_score", 0)),
            ),
            reverse=True,
        )
        grouped.append(
            {
                "business": business,
                "documents": docs,
                "top_star": int(docs[0].get("urgency_stars", 1)) if docs else 1,
            }
        )
    grouped.sort(key=lambda item: (int(item.get("top_star", 1)), item.get("business", "")), reverse=True)
    return grouped


def _build_interest_digest(messages: list[dict[str, str]], interests: list[str], max_items: int) -> dict[str, Any]:
    interest_terms = [item.strip().lower() for item in interests if item and item.strip()]
    if not interest_terms:
        interest_terms = ["需求", "上线", "风险", "客户", "复盘"]

    items: list[dict[str, Any]] = []
    dedupe_summary: set[str] = set()
    for message in messages:
        raw_text = message.get("text", "")
        summary_text = message.get("summary_text", "")
        base_text = summary_text or raw_text
        if not base_text:
            continue
        normalized_text = _normalize_interest_text(base_text)
        if _is_noise_interest_text(normalized_text):
            continue
        hit_terms = [term for term in interest_terms if term in normalized_text.lower()]
        if not hit_terms:
            continue
        signal_hits = sum(1 for keyword in INTEREST_SIGNAL_KEYWORDS if keyword in normalized_text.lower())
        urgency_hit = any(keyword in normalized_text.lower() for keyword in URGENCY_KEYWORDS)
        if len(hit_terms) < 2 and signal_hits <= 0 and not urgency_hit:
            continue
        rewritten = _rewrite_interest_summary(normalized_text, hit_terms=hit_terms)
        if not rewritten:
            continue
        summary = _trim(rewritten, 120)
        if summary in dedupe_summary:
            continue
        dedupe_summary.add(summary)
        score = len(hit_terms) * 30 + signal_hits * 8 + (20 if urgency_hit else 0)
        items.append(
            {
                "message_id": message.get("message_id", ""),
                "chat_id": message.get("chat_id", ""),
                "summary": summary,
                "hit_terms": hit_terms[:5],
                "message_url": _build_lark_message_url(
                    chat_id=str(message.get("chat_id", "")),
                    message_id=str(message.get("message_id", "")),
                ),
                "score": score,
                "create_time": message.get("create_time", ""),
            }
        )
    items.sort(key=lambda item: (int(item.get("score", 0)), str(item.get("create_time", ""))), reverse=True)
    return {
        "interests": interests,
        "items": items[:max_items],
    }


def _normalize_interest_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _is_noise_interest_text(text: str) -> bool:
    lowered = text.lower()
    if len(lowered) < 6:
        return True
    if _looks_like_bot_prompt(lowered):
        return True
    if "\ncommand:" in lowered:
        return True
    if lowered.startswith("command:"):
        return True
    if "要求如下" in lowered and "1." in lowered and "2." in lowered:
        return True
    return any(pattern in lowered for pattern in INTEREST_NOISE_PATTERNS)


def _looks_like_bot_prompt(lowered: str) -> bool:
    if lowered.startswith("@sofree") and ("请生成" in lowered or "要求如下" in lowered):
        return True
    if lowered.startswith("@_user_") and ("可以从几个维度讨论" in lowered or "口味偏好" in lowered):
        return True
    if lowered.startswith("@") and "1." in lowered and "2." in lowered and "3." in lowered and len(lowered) > 80:
        return True
    return False


def _rewrite_interest_summary(text: str, hit_terms: list[str]) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"@\S+", "", value)
    value = re.sub(r"\{.*?json_card.*?\}", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""
    if "要求如下" in value:
        value = value.split("要求如下", 1)[0].strip("：:，, ")
    segments = [seg.strip() for seg in re.split(r"[。！？!?\n；;]", value) if seg.strip()]
    if not segments:
        return ""
    preferred = ""
    for seg in segments:
        lowered = seg.lower()
        if any(term in lowered for term in hit_terms):
            preferred = seg
            break
    if not preferred:
        preferred = segments[0]
    preferred = re.sub(r"^\d+\.\s*", "", preferred)
    preferred = re.sub(r"^(请|帮我|生成|安排)", "", preferred).strip()
    if not preferred:
        return ""
    return preferred


def _build_runtime_plan(schedule: dict[str, Any]) -> dict[str, Any]:
    cron_jobs: list[dict[str, Any]] = []
    cron_jobs.append(
        {
            "job_name": "assistant_weekly_brief",
            "enabled": bool(schedule.get("weekly_enabled", True)),
            "cron": str(schedule.get("weekly_brief_cron", "0 9 * * MON")),
            "handler": "assistant build-personal-brief --online --output-format all",
        }
    )
    cron_jobs.append(
        {
            "job_name": "assistant_nightly_interest_digest",
            "enabled": bool(schedule.get("nightly_enabled", True)),
            "cron": str(schedule.get("nightly_interest_cron", "0 21 * * *")),
            "handler": "assistant build-personal-brief --online --output-format card",
        }
    )
    return {
        "cron_jobs": cron_jobs,
        "manual_run": {
            "enabled": bool(schedule.get("manual_supported", True)),
            "command": "assistant build-personal-brief --online --output-format all",
        },
        "auth_requirement": "需要使用用户 OAuth 凭据，只读取用户有权限的数据。",
    }


def _build_lark_message_url(chat_id: str, message_id: str) -> str:
    normalized_chat_id = str(chat_id or "").strip()
    if not normalized_chat_id:
        return ""
    # Keep chat-level deep link only. openMessageId often falls back to browser and may not resolve.
    return f"https://applink.feishu.cn/client/chat/{normalized_chat_id}"


def _is_recent(raw: str) -> bool:
    value = str(raw or "").strip()
    if not value:
        return False
    if value.isdigit():
        timestamp = int(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp // 1000
        now_ts = int(datetime.now(timezone.utc).timestamp())
        return (now_ts - timestamp) <= 7 * 24 * 3600
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).days <= 7


def _trim(text: str, max_len: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def _to_markdown(
    documents: list[dict[str, Any]],
    grouped_documents: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    lines = ["# 个人助理聚合简报", ""]
    role = str(profile.get("role") or "").strip()
    persona = str(profile.get("persona") or "").strip()
    if role or persona:
        lines.append(f"> 画像：{role or '未设置角色'} / {persona or '未设置形象'}")
        lines.append("")

    lines.append("## 业务分类")
    if not grouped_documents:
        lines.append("- 暂无业务分类数据")
    for group in grouped_documents:
        business = str(group.get("business") or "General")
        lines.append(f"### {business}")
        for item in group.get("documents", [])[:5]:
            title = f"[{item['title']}]({item['url']})" if item.get("url") else item["title"]
            lines.append(
                f"- {'★' * int(item['urgency_stars'])}{'☆' * (5 - int(item['urgency_stars']))} "
                f"[{item['doc_type']}] {title}"
            )
            lines.append(f"  - 描述: {item['summary'] or '无'}")
            if item["todo_items"]:
                lines.append(f"  - 待办: {item['todo_items'][0]}")
    lines.append("")
    lines.append("## 文档优先级")
    if not documents:
        lines.append("- 暂无可推荐文档")
        return "\n".join(lines)
    for index, item in enumerate(documents, start=1):
        title = f"[{item['title']}]({item['url']})" if item.get("url") else item["title"]
        lines.append(
            f"{index}. {'★' * int(item['urgency_stars'])}{'☆' * (5 - int(item['urgency_stars']))} "
            f"[{item['priority'].upper()}] {title} "
            f"(紧急:{item['urgency_score']} 推荐:{item['recommend_score']})"
        )
        for reason in item["reasons"][:2]:
            lines.append(f"   - {reason}")
        if item["todo_items"]:
            lines.append(f"   - 待办: {item['todo_items'][0]}")
        if item["related_messages"]:
            lines.append(f"   - 相关消息: {item['related_messages'][0]['summary']}")
        if item["related_knowledge"]:
            lines.append(f"   - 知识建议: {item['related_knowledge'][0]['title']}")
    return "\n".join(lines)


def _to_card(documents: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    content_lines: list[str] = []
    for item in documents[:5]:
        title = (
            f"[{_safe_markdown_link_text(item['title'])}]({item['url']})"
            if item.get("url")
            else item["title"]
        )
        content_lines.append(
            f"- {'★' * int(item['urgency_stars'])}{'☆' * (5 - int(item['urgency_stars']))} "
            f"**{item['priority'].upper()}** [{item['business']}] {title} "
            f"(紧急:{item['urgency_score']} 推荐:{item['recommend_score']})"
        )
    if not content_lines:
        content_lines = ["- 暂无可推荐文档"]
    role = str(profile.get("role") or "").strip()
    persona = str(profile.get("persona") or "").strip()
    subtitle = f"{role} / {persona}".strip(" /")
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "turquoise",
            "title": {"tag": "plain_text", "content": "个人助理聚合建议"},
            "subtitle": {"tag": "plain_text", "content": subtitle or "业务聚合"},
        },
        "elements": [{"tag": "markdown", "content": "\n".join(content_lines)}],
    }


def _to_interest_card(interest_digest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    lines: list[str] = []
    for item in interest_digest.get("items", [])[:5]:
        summary = str(item.get("summary", "") or "").strip()
        message_url = str(item.get("message_url", "") or "").strip()
        message_id = str(item.get("message_id", "") or "").strip()
        hit_terms = item.get("hit_terms", [])
        hit_label = ""
        if isinstance(hit_terms, list) and hit_terms:
            hit_label = f"（命中: {'+'.join(str(term) for term in hit_terms[:3])}）"
        id_label = f" [msg:{message_id[:12]}]" if message_id else ""
        if message_url:
            lines.append(f"- [{_safe_markdown_link_text(summary)}]({message_url}){hit_label}{id_label}")
        else:
            lines.append(f"- {summary}{hit_label}{id_label}")
    if not lines:
        lines.append("- 暂无命中兴趣的群聊内容")
    interests = profile.get("interests", [])
    interests_text = "、".join(interests[:4]) if interests else "未设置兴趣词"
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": "群聊兴趣消息汇总"},
            "subtitle": {"tag": "plain_text", "content": f"兴趣词: {interests_text}"},
        },
        "elements": [{"tag": "markdown", "content": "\n".join(lines)}],
    }


def _safe_markdown_link_text(text: str) -> str:
    value = str(text or "").strip()
    value = value.replace("[", "【").replace("]", "】")
    value = value.replace("(", "（").replace(")", "）")
    return value
