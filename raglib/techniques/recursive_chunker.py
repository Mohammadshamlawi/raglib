"""RecursiveChunker: Recursive text chunking with hierarchical splitting."""

from typing import Union

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Document


@TechniqueRegistry.register
class RecursiveChunker(RAGTechnique):
    """Recursive chunking technique with hierarchical text splitting.

    Splits text using a hierarchy of separators, starting with the most
    natural boundaries and recursively splitting larger chunks until
    size constraints are met.
    """

    meta = TechniqueMeta(
        name="recursive_chunker",
        category="chunking",
        description="Recursive chunking with hierarchical text splitting"
    )

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 50,
        separators: list[str] = None
    ):
        """Initialize RecursiveChunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between adjacent chunks
            separators: List of separators in order of preference
        """
        super().__init__(self.meta)
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Default separator hierarchy
        if separators is None:
            self.separators = [
                "\n\n",      # Paragraphs
                "\n",        # Lines
                " ",         # Words
                ".",         # Sentences (by period)
                ",",         # Clauses
                ""           # Characters (last resort)
            ]
        else:
            self.separators = separators

    def _split_by_characters(self, text: str, document_id: str) -> list[dict]:
        """Split text by characters as last resort."""
        if not text.strip():
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'start_idx': start,
                    'end_idx': end,
                    'document_id': document_id
                })
            
            # Calculate next start position with overlap, ensuring progress
            if self.overlap > 0 and self.overlap < len(chunk_text):
                start = end - self.overlap
            else:
                start = end
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks

    def _split_text(self, text: str, document_id: str) -> list[dict]:
        """Split text using simple approach."""
        if not text.strip():
            return []
            
        # If text fits in chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'start_idx': 0,
                'end_idx': len(text),
                'document_id': document_id
            }]

        # Try paragraph splitting first
        if "\n\n" in text:
            paragraphs = text.split("\n\n")
            chunks = []
            current_chunk = ""
            start_pos = 0
            
            for para in paragraphs:
                # Add paragraph separator back
                para_with_sep = para + "\n\n"
                
                if len(current_chunk + para_with_sep) <= self.chunk_size:
                    current_chunk += para_with_sep
                else:
                    # Finalize current chunk
                    if current_chunk.strip():
                        chunks.append({
                            'text': current_chunk.rstrip(),
                            'start_idx': start_pos,
                            'end_idx': start_pos + len(current_chunk.rstrip()),
                            'document_id': document_id
                        })
                        start_pos += len(current_chunk)
                    
                    # Start new chunk
                    if len(para_with_sep) <= self.chunk_size:
                        current_chunk = para_with_sep
                    else:
                        # Para too large, split by characters
                        char_chunks = self._split_by_characters(para, document_id)
                        for chunk in char_chunks:
                            chunk['start_idx'] += start_pos
                            chunk['end_idx'] += start_pos
                        chunks.extend(char_chunks)
                        start_pos += len(para_with_sep)
                        current_chunk = ""
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.rstrip(),
                    'start_idx': start_pos,
                    'end_idx': start_pos + len(current_chunk.rstrip()),
                    'document_id': document_id
                })
            
            return chunks
        
        # Fallback to character splitting
        return self._split_by_characters(text, document_id)

    def apply(self, document: Union[Document, str], *args, **kwargs) -> TechniqueResult:
        """Apply recursive chunking to document.

        Args:
            document: Document object or plain text string
            chunk_size: Override default chunk size (optional)
            overlap: Override default overlap (optional)
            separators: Override default separators (optional)

        Returns:
            TechniqueResult with chunks in payload
        """
        # Extract parameters
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        overlap = kwargs.get('overlap', self.overlap)
        separators = kwargs.get('separators', self.separators)

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

        # Temporarily update instance variables
        original_chunk_size = self.chunk_size
        original_overlap = self.overlap
        original_separators = self.separators
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators

        try:
            # Split the text
            chunk_data = self._split_text(text, document_id)

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
                    embedding=None,
                    meta={
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "chunk_index": chunk_index,
                        "method": "recursive",
                        "separators": separators,
                        **doc_meta
                    }
                )
                chunks.append(chunk)

            return TechniqueResult(
                success=True,
                payload={"chunks": chunks},
                meta={
                    "total_chunks": len(chunks),
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "separators": separators,
                    "original_length": len(text)
                }
            )
        finally:
            # Restore original values
            self.chunk_size = original_chunk_size
            self.overlap = original_overlap
            self.separators = original_separators