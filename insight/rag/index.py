from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from ..shared.models import RagHit
from .embed import Embedder


class VectorKnowledgeStore:
    def __init__(self, root_dir: str | Path, *, embed_model: str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.faiss"
        self.meta_path = self.root / "meta.json"
        self.embedder = Embedder(model_name=embed_model, normalize=True)
        self._lock = RLock()
        self._faiss = None

    def upsert_knowledge(
        self,
        *,
        knowledge_id: str,
        card_id: str,
        title: str,
        topic_focus: str,
        summary: str,
        times: str,
        locations: str,
        participants: list[str],
    ) -> bool:
        text = self._build_text(
            title=title,
            topic_focus=topic_focus,
            summary=summary,
            times=times,
            locations=locations,
            participants=participants,
        )
        if not text.strip():
            return False
        with self._lock:
            meta = self._load_meta()
            if any(str(item.get("knowledge_id", "")) == knowledge_id for item in meta):
                return True
            vec = self.embedder.encode([text])
            if vec.size == 0:
                return False
            index = self._load_or_create_index(vec.shape[1])
            index.add(vec)
            meta.append(
                {
                    "knowledge_id": knowledge_id,
                    "card_id": card_id,
                    "title": title,
                    "topic_focus": topic_focus,
                    "summary": summary,
                    "times": times,
                    "locations": locations,
                    "participants": participants,
                    "text": text,
                }
            )
            self._save_index(index)
            self._save_meta(meta)
        return True

    def search(self, *, query: str, top_k: int, min_score: float = 0.2) -> list[RagHit]:
        text = str(query or "").strip()
        if not text:
            return []
        with self._lock:
            meta = self._load_meta()
            if not meta or not self.index_path.exists():
                return []
            index = self._load_index()
            if index is None:
                return []
            q = self.embedder.encode([text])
            if q.size == 0:
                return []
            k = max(1, min(int(top_k or 1), len(meta)))
            scores, ids = index.search(q, k)
            hits: list[RagHit] = []
            for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
                if idx < 0 or idx >= len(meta):
                    continue
                score_val = float(score)
                if score_val < float(min_score):
                    continue
                row = meta[idx]
                hits.append(
                    RagHit(
                        knowledge_id=str(row.get("knowledge_id", "")),
                        card_id=str(row.get("card_id", "")),
                        score=score_val,
                        text=str(row.get("text", "")),
                        title=str(row.get("title", "")),
                        summary=str(row.get("summary", "")),
                        evidence=[],
                        tags=[],
                    )
                )
            return hits

    def _load_or_create_index(self, dim: int):
        index = self._load_index()
        if index is not None:
            return index
        import faiss

        return faiss.IndexFlatIP(dim)

    def _load_index(self):
        if self._faiss is not None:
            return self._faiss
        if not self.index_path.exists():
            return None
        import faiss

        self._faiss = faiss.read_index(str(self.index_path))
        return self._faiss

    def _save_index(self, index) -> None:
        import faiss

        faiss.write_index(index, str(self.index_path))
        self._faiss = index

    def _load_meta(self) -> list[dict[str, Any]]:
        if not self.meta_path.exists():
            return []
        try:
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _save_meta(self, rows: list[dict[str, Any]]) -> None:
        self.meta_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _build_text(*, title: str, topic_focus: str, summary: str, times: str, locations: str, participants: list[str]) -> str:
        people = ", ".join(str(item).strip() for item in participants if str(item).strip())
        return (
            f"title: {str(title or '').strip()}\n"
            f"topic_focus: {str(topic_focus or '').strip()}\n"
            f"summary: {str(summary or '').strip()}\n"
            f"times: {str(times or '').strip()}\n"
            f"locations: {str(locations or '').strip()}\n"
            f"participants: {people}"
        ).strip()


class VectorObserveStore:
    def __init__(self, root_dir: str | Path, *, embed_model: str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.faiss"
        self.meta_path = self.root / "meta.json"
        self.embedder = Embedder(model_name=embed_model, normalize=True)
        self._lock = RLock()
        self._faiss = None

    def upsert_observe(
        self,
        *,
        observe_id: str,
        title: str,
        topic_focus: str,
        summary: str,
        times: str,
        locations: str,
        participants: list[str],
    ) -> bool:
        text = self._build_text(
            title=title,
            topic_focus=topic_focus,
            summary=summary,
            times=times,
            locations=locations,
            participants=participants,
        )
        if not text.strip():
            return False
        with self._lock:
            meta = self._load_meta()
            vec = self.embedder.encode([text])
            if vec.size == 0:
                return False
            row = {
                "observe_id": observe_id,
                "title": title,
                "topic_focus": topic_focus,
                "summary": summary,
                "times": times,
                "locations": locations,
                "participants": participants,
                "text": text,
            }
            existing_idx = next((idx for idx, item in enumerate(meta) if str(item.get("observe_id", "")) == observe_id), None)
            if existing_idx is None:
                index = self._load_or_create_index(vec.shape[1])
                index.add(vec)
                meta.append(row)
                self._save_index(index)
                self._save_meta(meta)
                return True
            meta[existing_idx] = row
            dim = vec.shape[1]
            index = self._rebuild_index(meta, dim)
            self._save_index(index)
            self._save_meta(meta)
        return True

    def search(self, *, query: str, top_k: int, min_score: float = 0.2) -> list[dict[str, Any]]:
        text = str(query or "").strip()
        if not text:
            return []
        with self._lock:
            meta = self._load_meta()
            if not meta or not self.index_path.exists():
                return []
            index = self._load_index()
            if index is None:
                return []
            q = self.embedder.encode([text])
            if q.size == 0:
                return []
            k = max(1, min(int(top_k or 1), len(meta)))
            scores, ids = index.search(q, k)
            out: list[dict[str, Any]] = []
            for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
                if idx < 0 or idx >= len(meta):
                    continue
                score_val = float(score)
                if score_val < float(min_score):
                    continue
                row = meta[idx]
                out.append(
                    {
                        "observe_id": str(row.get("observe_id", "")),
                        "topic": str(row.get("title", "")),
                        "summary": str(row.get("summary", "")),
                        "text": str(row.get("text", "")),
                        "score": score_val,
                    }
                )
            return out

    def _rebuild_index(self, rows: list[dict[str, Any]], dim: int):
        import faiss

        index = faiss.IndexFlatIP(dim)
        texts = [str(item.get("text", "")).strip() for item in rows]
        vectors = self.embedder.encode(texts)
        if vectors.size > 0:
            index.add(vectors)
        return index

    def _load_or_create_index(self, dim: int):
        index = self._load_index()
        if index is not None:
            return index
        import faiss

        return faiss.IndexFlatIP(dim)

    def _load_index(self):
        if self._faiss is not None:
            return self._faiss
        if not self.index_path.exists():
            return None
        import faiss

        self._faiss = faiss.read_index(str(self.index_path))
        return self._faiss

    def _save_index(self, index) -> None:
        import faiss

        faiss.write_index(index, str(self.index_path))
        self._faiss = index

    def _load_meta(self) -> list[dict[str, Any]]:
        if not self.meta_path.exists():
            return []
        try:
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _save_meta(self, rows: list[dict[str, Any]]) -> None:
        self.meta_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _build_text(*, title: str, topic_focus: str, summary: str, times: str, locations: str, participants: list[str]) -> str:
        people = ", ".join(str(item).strip() for item in participants if str(item).strip())
        return (
            f"title: {str(title or '').strip()}\n"
            f"topic_focus: {str(topic_focus or '').strip()}\n"
            f"summary: {str(summary or '').strip()}\n"
            f"times: {str(times or '').strip()}\n"
            f"locations: {str(locations or '').strip()}\n"
            f"participants: {people}"
        ).strip()
