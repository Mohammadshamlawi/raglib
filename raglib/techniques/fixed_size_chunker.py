"""FixedSizeChunker: Fixed-size text chunking with overlap support."""

from typing import Union

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Document


@TechniqueRegistry.register
class FixedSizeChunker(RAGTechnique):
    """Fixed-size text chunking technique with configurable overlap.
    
    Splits text into chunks of specified size with optional overlap between chunks.
    Each chunk maintains character-level position information.
    """

    meta = TechniqueMeta(
        name="Fixed-Size Chunking",
        category="chunking",
        description="Fixed-size text chunking with overlap support"
    )

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize FixedSizeChunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between adjacent chunks
        """
        super().__init__(self.meta)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def apply(self, document: Union[Document, str], *args, **kwargs) -> TechniqueResult:
        """Apply fixed-size chunking to document.
        
        Args:
            document: Document object or plain text string
            chunk_size: Override default chunk size (optional)
            overlap: Override default overlap (optional)
            
        Returns:
            TechniqueResult with chunks in payload
        """
        # Extract parameters
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        overlap = kwargs.get('overlap', self.overlap)

        # Validate parameters
        if chunk_size <= 0:
            return TechniqueResult(
                success=False,
                payload={"error": "chunk_size must be positive"},
                meta={"chunk_size": chunk_size, "overlap": overlap}
            )

        if overlap < 0 or overlap >= chunk_size:
            return TechniqueResult(
                success=False,
                payload={"error": "overlap must be >= 0 and < chunk_size"},
                meta={"chunk_size": chunk_size, "overlap": overlap}
            )

        # Extract text and document_id
        if isinstance(document, Document):
            text = document.text
            document_id = document.id
            doc_meta = document.meta
        else:
            text = str(document)
            document_id = "unknown"
            doc_meta = {}

        # Calculate step size
        step = chunk_size - overlap

        chunks = []
        chunk_index = 0
        start_pos = 0

        while start_pos < len(text):
            # Calculate end position
            end_pos = min(start_pos + chunk_size, len(text))

            # Extract chunk text
            chunk_text = text[start_pos:end_pos]

            # Create chunk
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                start_idx=start_pos,
                end_idx=end_pos,
                embedding=None,  # No embeddings for basic chunker
                meta={
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "chunk_index": chunk_index,
                    "method": "fixed_size",
                    **doc_meta
                }
            )

            chunks.append(chunk)

            # Move to next chunk position
            chunk_index += 1
            start_pos += step

            # If we've reached the end exactly, break
            if end_pos == len(text):
                break

        return TechniqueResult(
            success=True,
            payload={"chunks": chunks},
            meta={
                "total_chunks": len(chunks),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "original_length": len(text)
            }
        )
