from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Document:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    start_idx: int
    end_idx: int
    embedding: Optional[list[float]] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    doc_id: str
    score: float
    chunk: Optional[Chunk] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagResult:
    """Result from a RAG technique operation."""
    documents: list[Document] = field(default_factory=list)
    response: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
