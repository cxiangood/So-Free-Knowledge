"""Assistant domain helpers."""

from .collector import collect_personal_inputs_online
from .features import (
    extract_keywords,
    normalize_doc,
    normalize_knowledge,
    normalize_message,
    normalize_profile,
    normalize_schedule,
)
from .models import (
    AccessRecord,
    BusinessTrack,
    DocumentCandidate,
    DocumentRecord,
    DocumentRetrievalDebug,
    DualTowerConfig,
    InterestDigestItem,
    KnowledgeItem,
    MessageRecord,
    RankedDocument,
    RelatedKnowledge,
    RelatedMessage,
    ScheduleConfig,
    UserProfile,
)
from .profile import (
    assistant_profile_default_path,
    build_profile_overrides,
    build_schedule_overrides,
    load_assistant_profile_config,
)
from .dual_tower_dataset import build_user_tower_text_with_topics, build_weak_supervision_samples
from .ranker import score_doc_recommend, score_doc_urgency
from .renderer import to_card, to_interest_card, to_markdown
from .retriever import build_document_candidate, build_interest_digest, rank_related_knowledge, rank_related_messages
from .service import build_personal_brief_command_result, recommend_command_result, resolve_push_target
from .training import export_dual_tower_samples, load_dual_tower_samples, train_dual_tower_baseline
from .two_tower import build_document_tower_text, build_dual_tower_debug_payload, build_user_tower_text

__all__ = [
    "AccessRecord",
    "BusinessTrack",
    "DocumentCandidate",
    "DocumentRecord",
    "DocumentRetrievalDebug",
    "DualTowerConfig",
    "InterestDigestItem",
    "KnowledgeItem",
    "MessageRecord",
    "RankedDocument",
    "RelatedKnowledge",
    "RelatedMessage",
    "ScheduleConfig",
    "UserProfile",
    "assistant_profile_default_path",
    "build_document_candidate",
    "build_document_tower_text",
    "build_dual_tower_debug_payload",
    "build_interest_digest",
    "build_personal_brief_command_result",
    "build_profile_overrides",
    "build_schedule_overrides",
    "build_user_tower_text_with_topics",
    "build_weak_supervision_samples",
    "build_user_tower_text",
    "collect_personal_inputs_online",
    "extract_keywords",
    "export_dual_tower_samples",
    "load_dual_tower_samples",
    "load_assistant_profile_config",
    "normalize_doc",
    "normalize_knowledge",
    "normalize_message",
    "normalize_profile",
    "normalize_schedule",
    "rank_related_knowledge",
    "rank_related_messages",
    "recommend_command_result",
    "resolve_push_target",
    "score_doc_recommend",
    "score_doc_urgency",
    "to_card",
    "to_interest_card",
    "to_markdown",
    "train_dual_tower_baseline",
]
