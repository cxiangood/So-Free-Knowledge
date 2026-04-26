"""SoFree Knowledge standalone CLI package."""

__version__ = "0.1.0"

from .confused_detector import (
    build_confused_judge_prompt,
    detect_confused_candidates,
    format_inline_explanation,
    parse_confused_judgement,
)
from .lingo_context import (
    build_lingo_judge_prompt,
    extract_keyword_contexts,
    parse_lingo_judgements,
    publishable_lingo_judgements,
)

__all__ = [
    "build_confused_judge_prompt",
    "build_lingo_judge_prompt",
    "detect_confused_candidates",
    "extract_keyword_contexts",
    "format_inline_explanation",
    "parse_confused_judgement",
    "parse_lingo_judgements",
    "publishable_lingo_judgements",
]
