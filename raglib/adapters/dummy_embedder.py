"""DummyEmbedder: deterministic, sha256-based embeddings for testing."""

from typing import Sequence, List
import hashlib
import math
import re

from .base import EmbedderAdapter

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> Sequence[str]:
    return _WORD_RE.findall(text.lower())


def _embed_text(text: str, dim: int) -> List[float]:
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        for i in range(dim):
            # cycle through digest bytes deterministically
            vec[i] += digest[i % len(digest)]
    # normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return [0.0] * dim
    return [float(x) / norm for x in vec]


class DummyEmbedder(EmbedderAdapter):
    """Simple deterministic embedder.

    Args:
        dim: embedding dimensionality (default 16)
    """
    def __init__(self, dim: int = 16):
        self.dim = int(dim)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [_embed_text(str(t), self.dim) for t in texts]
