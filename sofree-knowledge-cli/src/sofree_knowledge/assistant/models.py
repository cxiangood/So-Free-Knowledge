from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class UserProfile:
    persona: str = ""
    role: str = ""
    interests: list[str] = field(default_factory=list)
    business_tracks: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class BusinessTrack:
    id: str = ""
    name: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ScheduleConfig:
    mode: str = "scheduled"
    timezone: str = "Asia/Shanghai"
    weekly_brief_cron: str = "0 9 * * MON"
    nightly_interest_cron: str = "0 21 * * *"
    weekly_enabled: bool = True
    nightly_enabled: bool = True


@dataclass(slots=True)
class MessageRecord:
    message_id: str = ""
    chat_id: str = ""
    sender_name: str = ""
    text: str = ""
    summary_text: str = ""
    create_time: str = ""
    message_url: str = ""


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str = ""
    title: str = ""
    summary: str = ""
    url: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class AccessRecord:
    doc_id: str = ""
    user_id: str = ""
    action: str = ""
    count: int = 1


@dataclass(slots=True)
class KnowledgeItem:
    id: str = ""
    title: str = ""
    content: str = ""


@dataclass(slots=True)
class RelatedMessage:
    message_id: str = ""
    chat_id: str = ""
    summary: str = ""
    score: int = 0
    create_time: str = ""


@dataclass(slots=True)
class RelatedKnowledge:
    id: str = ""
    title: str = ""
    summary: str = ""
    score: int = 0


@dataclass(slots=True)
class DocumentRetrievalDebug:
    strategy: str = "rule_based"
    dual_tower_score: float = 0.0
    trained_dual_tower_score: float = 0.0
    lexical_overlap: int = 0
    related_message_count: int = 0
    related_knowledge_count: int = 0
    user_tower_text: str = ""
    content_tower_text: str = ""
    dual_tower_ready: bool = False
    model_applied: bool = False
    model_file: str = ""


@dataclass(slots=True)
class DocumentCandidate:
    doc_id: str = ""
    doc_type: str = "file"
    business: str = "General"
    title_keywords: set[str] = field(default_factory=set)
    related_messages: list[RelatedMessage] = field(default_factory=list)
    related_knowledge: list[RelatedKnowledge] = field(default_factory=list)
    retrieval_debug: DocumentRetrievalDebug = field(default_factory=DocumentRetrievalDebug)


@dataclass(slots=True)
class DualTowerConfig:
    enabled: bool = True
    embedding_model: str = ""
    model_file: str = ""
    top_k: int = 50
    min_score: float = 0.0
    user_fields: list[str] = field(default_factory=lambda: ["role", "persona", "business_tracks", "interests"])
    content_fields: list[str] = field(default_factory=lambda: ["title", "summary", "business", "doc_type"])


@dataclass(slots=True)
class RankedDocument:
    doc_id: str = ""
    title: str = ""
    business: str = "General"
    recommend_score: int = 0
    urgency_score: int = 0
    priority: str = "normal"


@dataclass(slots=True)
class InterestDigestItem:
    message_id: str = ""
    chat_id: str = ""
    summary: str = ""
    score: int = 0
    message_url: str = ""
