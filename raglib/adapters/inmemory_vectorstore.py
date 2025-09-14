"""In-memory vector store adapter.

Very small, dependency-free vector store for tests and examples. Stores:
- ids: list of string ids
- vectors: list of list[float]
- metadata: list of dict

Search uses cosine similarity over stored vectors (linear scan).
"""
import math
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

from .base import VectorStoreAdapter


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return float(dot / denom) if denom > 0 else 0.0


class InMemoryVectorStore(VectorStoreAdapter):
    def __init__(self):
        self._ids: List[str] = []
        self._vectors: List[List[float]] = []
        self._metadata: List[Optional[Dict]] = []

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metadata: Optional[Sequence[Dict]] = None) -> None:
        if metadata is None:
            metadata = [None] * len(ids)
        for i, _id in enumerate(ids):
            self._ids.append(_id)
            self._vectors.append(list(vectors[i]))
            self._metadata.append(metadata[i] if i < len(metadata) else None)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[Tuple[str, float, Optional[Dict]]]:
        scores = []
        for i, vec in enumerate(self._vectors):
            sc = _cosine(query_vector, vec)
            scores.append((self._ids[i], sc, self._metadata[i]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
