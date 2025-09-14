from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple


class EmbedderAdapter(ABC):
    """Abstract interface for an embedder."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return list of vector embeddings for the provided texts."""
        raise NotImplementedError


class VectorStoreAdapter(ABC):
    """Abstract interface for a vector store."""

    @abstractmethod
    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metadata: Optional[Sequence[Dict[str, Any]]]=None) -> None:
        """Add vectors and associated metadata to the store."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Search the store for nearest vectors.

        Returns a list of tuples (id, score, metadata) ordered by score descending.
        """
        raise NotImplementedError


class LLMAdapter(ABC):
    """Abstract interface for an LLM generator."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt and return it."""
        raise NotImplementedError
