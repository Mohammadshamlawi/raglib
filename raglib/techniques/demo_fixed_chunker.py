"""Demo fixed-size chunker technique.

This is a simple, dependency-free chunker that splits an input string into
fixed-size character chunks with optional overlap. It demonstrates the one-file-per-technique
rule and the required `apply` signature.
"""
from typing import Any, List
from dataclasses import asdict

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..schemas import Chunk, Document
from ..registry import TechniqueRegistry


@TechniqueRegistry.register
class DemoFixedChunker(RAGTechnique):
    meta = TechniqueMeta(
        name="demo_fixed_chunker",
        category="chunking",
        description="Simple fixed-size character chunker for demos and tests."
    )

    def __init__(self):
        super().__init__(self.meta)

    def apply(self, document: Any, chunk_size: int = 200, overlap: int = 0) -> TechniqueResult:
        """Split the document text into chunks.

        Args:
            document: either a Document dataclass instance or a plain string.
            chunk_size: max characters per chunk.
            overlap: number of characters to overlap between chunks.

        Returns:
            TechniqueResult with payload { "chunks": List[Chunk] }.
        """
        # Accept both Document or raw string for convenience in tests/examples
        text = document.text if hasattr(document, "text") else str(document)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        step = chunk_size - overlap if chunk_size > overlap else chunk_size
        chunks: List[Chunk] = []
        start = 0
        idx = 0
        doc_id = getattr(document, "id", "doc_0")
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunk = Chunk(
                id=f"{doc_id}_chunk_{idx}",
                document_id=doc_id,
                text=chunk_text,
                start_idx=start,
                end_idx=end,
                embedding=None,
                meta={}
            )
            chunks.append(chunk)
            idx += 1
            start += step
        return TechniqueResult(success=True, payload={"chunks": chunks})
