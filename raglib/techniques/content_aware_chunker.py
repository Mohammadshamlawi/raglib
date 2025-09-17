"""ContentAwareChunker: Content-aware chunking with text structure."""

import re
from typing import Union

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Document


@TechniqueRegistry.register
class ContentAwareChunker(RAGTechnique):
    """Content-aware chunking technique that respects text structure.

    Splits text at natural boundaries (paragraphs, sentences, headings) while
    maintaining context and respecting maximum chunk size limits.
    """

    meta = TechniqueMeta(
        name="content_aware_chunker",
        category="chunking",
        description=(
            "Content-aware chunking that respects text structure "
            "and natural boundaries"
        )
    )

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 0
    ):
        """Initialize ContentAwareChunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
            overlap: Number of characters to overlap between adjacent chunks
        """
        super().__init__(self.meta)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def _find_natural_boundaries(self, text: str) -> list[int]:
        """Find natural text boundaries (paragraphs, sentences, etc.)."""
        boundaries = [0]  # Start of text

        # Find paragraph boundaries (double newlines)
        paragraph_pattern = r'\n\s*\n'
        for match in re.finditer(paragraph_pattern, text):
            boundaries.append(match.end())

        # Find heading boundaries (lines starting with #, or all caps lines)
        heading_pattern = r'\n(#{1,6}\s+.+|[A-Z\s]{5,})\n'
        for match in re.finditer(heading_pattern, text):
            boundaries.append(match.start() + 1)

        # Find sentence boundaries
        sentence_pattern = r'[.!?]+\s+'
        for match in re.finditer(sentence_pattern, text):
            boundaries.append(match.end())

        # Add end of text
        boundaries.append(len(text))

        # Remove duplicates and sort
        boundaries = sorted(set(boundaries))

        return boundaries

    def _create_chunks_from_boundaries(
        self, text: str, boundaries: list[int], document_id: str
    ) -> list[dict]:
        """Create chunks respecting natural boundaries and size constraints."""
        chunks = []
        
        # Use a simpler approach: try to respect boundaries, but fall back to fixed-size
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(text):
            # Find the best boundary within max_chunk_size
            best_end = min(current_pos + self.max_chunk_size, len(text))
            
            # Look for a natural boundary before max_chunk_size
            for boundary in boundaries:
                if current_pos < boundary <= current_pos + self.max_chunk_size:
                    best_end = boundary
            
            # Extract chunk text
            chunk_text = text[current_pos:best_end].strip()
            
            # Only create chunk if it meets minimum size (except for last chunk)
            if len(chunk_text) >= self.min_chunk_size or best_end == len(text):
                if chunk_text:  # Don't create empty chunks
                    chunks.append({
                        'text': chunk_text,
                        'start_idx': current_pos,
                        'end_idx': best_end,
                        'document_id': document_id
                    })
            
            # Move to next position with overlap
            step = max(1, self.max_chunk_size - self.overlap)  # Ensure progress
            current_pos += step
            
            # Safety check to prevent infinite loops
            if current_pos >= len(text) or step <= 0:
                break
            
            chunk_index += 1
            # Additional safety: limit number of chunks
            if chunk_index > 10000:  # Reasonable upper bound
                break
        
        return chunks

    def apply(self, document: Union[Document, str], *args, **kwargs) -> TechniqueResult:
        """Apply content-aware chunking to document.
        
        Args:
            document: Document object or plain text string
            max_chunk_size: Override default max chunk size (optional)
            min_chunk_size: Override default min chunk size (optional)
            overlap: Override default overlap (optional)
            
        Returns:
            TechniqueResult with chunks in payload
        """
        # Extract parameters
        max_chunk_size = kwargs.get('max_chunk_size', self.max_chunk_size)
        min_chunk_size = kwargs.get('min_chunk_size', self.min_chunk_size)
        overlap = kwargs.get('overlap', self.overlap)

        # Adjust overlap if it's too large
        if overlap >= max_chunk_size:
            overlap = max(0, max_chunk_size // 4)  # Use 25% of chunk size as overlap

        # Validate parameters
        if max_chunk_size <= 0:
            return TechniqueResult(
                success=False,
                payload={"error": "max_chunk_size must be positive"},
                meta={
                    "max_chunk_size": max_chunk_size,
                    "min_chunk_size": min_chunk_size,
                    "overlap": overlap
                }
            )

        if min_chunk_size <= 0 or min_chunk_size >= max_chunk_size:
            return TechniqueResult(
                success=False,
                payload={
                    "error": "min_chunk_size must be positive and < max_chunk_size"
                },
                meta={
                    "max_chunk_size": max_chunk_size,
                    "min_chunk_size": min_chunk_size,
                    "overlap": overlap
                }
            )

        if overlap < 0:
            return TechniqueResult(
                success=False,
                payload={"error": "overlap must be >= 0"},
                meta={
                    "max_chunk_size": max_chunk_size,
                    "min_chunk_size": min_chunk_size,
                    "overlap": overlap
                }
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

        # Find natural boundaries in the text
        boundaries = self._find_natural_boundaries(text)

        # Create chunks respecting boundaries and size constraints
        chunk_data = self._create_chunks_from_boundaries(text, boundaries, document_id)

        # Convert to Chunk objects
        chunks = []
        for chunk_index, chunk_info in enumerate(chunk_data):
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=chunk_info['text'],
                start_idx=chunk_info['start_idx'],
                end_idx=chunk_info['end_idx'],
                embedding=None,  # No embeddings for basic chunker
                meta={
                    "max_chunk_size": max_chunk_size,
                    "min_chunk_size": min_chunk_size,
                    "overlap": overlap,
                    "chunk_index": chunk_index,
                    "method": "content_aware",
                    "boundaries_found": len(boundaries),
                    **doc_meta
                }
            )

            chunks.append(chunk)

        return TechniqueResult(
            success=True,
            payload={"chunks": chunks},
            meta={
                "total_chunks": len(chunks),
                "max_chunk_size": max_chunk_size,
                "min_chunk_size": min_chunk_size,
                "overlap": overlap,
                "natural_boundaries": len(boundaries),
                "original_length": len(text)
            }
        )
