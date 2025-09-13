from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TechniqueMeta:
    """Metadata describing a RAG technique."""
    name: str
    category: str
    description: str
    tags: Optional[Dict[str, Any]] = None


@dataclass
class TechniqueResult:
    """Uniform wrapper for technique outputs (recommended)."""
    success: bool
    payload: Any
    meta: Optional[Dict[str, Any]] = None


class RAGTechnique(ABC):
    """Abstract base class for RAG techniques.

    All concrete techniques must inherit from this class and must implement
    the single abstract method `apply(self, *args, **kwargs) -> Any`.
    """

    meta: TechniqueMeta

    def __init__(self, meta: TechniqueMeta):
        self.meta = meta

    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """Apply the RAG technique."""
        pass
