from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Embedder:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    normalize: bool = True
    _model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Prefer local HuggingFace cache first, then fall back to remote download.
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._get_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors)
        return vectors.astype(np.float32)
