"""SoFree Knowledge standalone CLI package."""

__version__ = "0.1.0"

from .confused_detector import (
    build_confused_judge_prompt,
    detect_confused_candidates,
    format_inline_explanation,
    parse_confused_judgement,
)
from .assistant_brief import build_personal_brief
from .assistant_online import collect_online_personal_inputs
from .lingo_context import (
    build_lingo_judge_prompt,
    extract_keyword_contexts,
    parse_lingo_judgements,
    publishable_lingo_judgements,
)
from .lingo_store import LingoStore

__all__ = [
    "build_confused_judge_prompt",
    "build_lingo_judge_prompt",
    "build_personal_brief",
    "collect_online_personal_inputs",
    "detect_confused_candidates",
    "extract_keyword_contexts",
    "format_inline_explanation",
    "LingoStore",
    "parse_confused_judgement",
    "parse_lingo_judgements",
    "publishable_lingo_judgements",
]
