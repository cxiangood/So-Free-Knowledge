from __future__ import annotations

from dataclasses import dataclass

from ..shared.models import RagHit

QUESTION_MARKERS = ("?", "？", "请问", "怎么", "如何", "吗", "是否", "能否", "为什么", "求助")


@dataclass(slots=True)
class ObserveAnswerResult:
    can_answer: bool
    answer: str = ""
    hits: list[RagHit] | None = None
    reason: str = ""


def is_question_by_rule(*, summary: str, problem: str, content: str) -> bool:
    text = f"{summary}\n{problem}\n{content}".lower()
    return any(marker in text for marker in QUESTION_MARKERS)


def try_answer_with_rag(query: str, hits: list[RagHit], min_hits: int = 1) -> ObserveAnswerResult:
    if not hits or len(hits) < max(1, int(min_hits or 1)):
        return ObserveAnswerResult(can_answer=False, reason="no_relevant_knowledge")
    top = hits[:3]
    lines: list[str] = [f"基于知识库检索，给你一个快速答复：{query}"]
    for idx, hit in enumerate(top, start=1):
        snippet = (hit.summary or hit.text or "").replace("\n", " ").strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(f"{idx}. {hit.title or '相关知识'}：{snippet}")
    return ObserveAnswerResult(can_answer=True, answer="\n".join(lines), hits=top)


__all__ = ["ObserveAnswerResult", "is_question_by_rule", "try_answer_with_rag"]
