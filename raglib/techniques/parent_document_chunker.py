"""ParentDocumentChunker: Parent document retrieval chunking technique."""

from typing import Union

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Document


@TechniqueRegistry.register
class ParentDocumentChunker(RAGTechnique):
    """Parent document retrieval chunking technique.

    Creates small chunks for retrieval while maintaining links to larger
    parent chunks for context. This enables precise retrieval with rich context.
    """

    meta = TechniqueMeta(
        name="parent_document_chunker",
        category="chunking",
        description="Parent document retrieval with small-to-large chunk mapping"
    )

    def __init__(
        self,
        small_chunk_size: int = 200,
        large_chunk_size: int = 1000,
        overlap: int = 50
    ):
        """Initialize ParentDocumentChunker.

        Args:
            small_chunk_size: Size of small chunks for retrieval
            large_chunk_size: Size of large chunks for context
            overlap: Overlap between chunks
        """
        super().__init__(self.meta)
        self.small_chunk_size = small_chunk_size
        self.large_chunk_size = large_chunk_size
        self.overlap = overlap

    def _create_large_chunks(self, text: str, document_id: str) -> list[dict]:
        """Create large parent chunks."""
        chunks = []
        step = self.large_chunk_size - self.overlap
        start_pos = 0
        chunk_index = 0

        while start_pos < len(text):
            end_pos = min(start_pos + self.large_chunk_size, len(text))
            chunk_text = text[start_pos:end_pos]

            chunks.append({
                'text': chunk_text,
                'start_idx': start_pos,
                'end_idx': end_pos,
                'chunk_index': chunk_index,
                'document_id': document_id,
                'type': 'large'
            })

            chunk_index += 1
            start_pos += step

            if end_pos == len(text):
                break

        return chunks

    def _create_small_chunks(self, text: str, document_id: str) -> list[dict]:
        """Create small chunks for retrieval."""
        chunks = []
        step = self.small_chunk_size - self.overlap
        start_pos = 0
        chunk_index = 0

        while start_pos < len(text):
            end_pos = min(start_pos + self.small_chunk_size, len(text))
            chunk_text = text[start_pos:end_pos]

            chunks.append({
                'text': chunk_text,
                'start_idx': start_pos,
                'end_idx': end_pos,
                'chunk_index': chunk_index,
                'document_id': document_id,
                'type': 'small'
            })

            chunk_index += 1
            start_pos += step

            if end_pos == len(text):
                break

        return chunks

    def _map_small_to_large(
        self, small_chunks: list[dict], large_chunks: list[dict]
    ) -> list[dict]:
        """Map small chunks to their parent large chunks."""
        for small_chunk in small_chunks:
            small_start = small_chunk['start_idx']
            small_end = small_chunk['end_idx']
            parent_chunks = []

            # Find all large chunks that overlap with this small chunk
            for large_chunk in large_chunks:
                large_start = large_chunk['start_idx']
                large_end = large_chunk['end_idx']

                # Check for overlap
                if (small_start < large_end and small_end > large_start):
                    parent_chunks.append(large_chunk['chunk_index'])

            small_chunk['parent_chunks'] = parent_chunks

        return small_chunks

    def apply(self, document: Union[Document, str], *args, **kwargs) -> TechniqueResult:
        """Apply parent document chunking to document.

        Args:
            document: Document object or plain text string
            small_chunk_size: Override default small chunk size (optional)
            large_chunk_size: Override default large chunk size (optional)
            overlap: Override default overlap (optional)

        Returns:
            TechniqueResult with small and large chunks in payload
        """
        # Extract parameters
        small_chunk_size = kwargs.get('small_chunk_size', self.small_chunk_size)
        large_chunk_size = kwargs.get('large_chunk_size', self.large_chunk_size)
        overlap = kwargs.get('overlap', self.overlap)

        # Validate parameters
        if small_chunk_size <= 0 or large_chunk_size <= 0:
            return TechniqueResult(
                success=False,
                payload={"error": "chunk sizes must be positive"},
                meta={
                    "small_chunk_size": small_chunk_size,
                    "large_chunk_size": large_chunk_size,
                    "overlap": overlap
                }
            )

        if small_chunk_size >= large_chunk_size:
            return TechniqueResult(
                success=False,
                payload={"error": "small_chunk_size must be < large_chunk_size"},
                meta={
                    "small_chunk_size": small_chunk_size,
                    "large_chunk_size": large_chunk_size,
                    "overlap": overlap
                }
            )

        if overlap < 0 or overlap >= small_chunk_size:
            return TechniqueResult(
                success=False,
                payload={"error": "overlap must be >= 0 and < small_chunk_size"},
                meta={
                    "small_chunk_size": small_chunk_size,
                    "large_chunk_size": large_chunk_size,
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

        # Create large and small chunks
        large_chunk_data = self._create_large_chunks(text, document_id)
        small_chunk_data = self._create_small_chunks(text, document_id)

        # Map small chunks to large chunks
        small_chunk_data = self._map_small_to_large(small_chunk_data, large_chunk_data)

        # Convert to Chunk objects
        small_chunks = []
        large_chunks = []

        # Create small chunks
        for chunk_info in small_chunk_data:
            chunk_id = f"{document_id}_small_{chunk_info['chunk_index']}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=chunk_info['text'],
                start_idx=chunk_info['start_idx'],
                end_idx=chunk_info['end_idx'],
                embedding=None,
                meta={
                    "small_chunk_size": small_chunk_size,
                    "large_chunk_size": large_chunk_size,
                    "overlap": overlap,
                    "chunk_index": chunk_info['chunk_index'],
                    "chunk_type": "small",
                    "parent_chunks": chunk_info['parent_chunks'],
                    "method": "parent_document",
                    **doc_meta
                }
            )
            small_chunks.append(chunk)

        # Create large chunks
        for chunk_info in large_chunk_data:
            chunk_id = f"{document_id}_large_{chunk_info['chunk_index']}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=chunk_info['text'],
                start_idx=chunk_info['start_idx'],
                end_idx=chunk_info['end_idx'],
                embedding=None,
                meta={
                    "small_chunk_size": small_chunk_size,
                    "large_chunk_size": large_chunk_size,
                    "overlap": overlap,
                    "chunk_index": chunk_info['chunk_index'],
                    "chunk_type": "large",
                    "method": "parent_document",
                    **doc_meta
                }
            )
            large_chunks.append(chunk)

        return TechniqueResult(
            success=True,
            payload={
                "small_chunks": small_chunks,
                "large_chunks": large_chunks,
                "chunks": small_chunks  # Default for compatibility
            },
            meta={
                "total_small_chunks": len(small_chunks),
                "total_large_chunks": len(large_chunks),
                "small_chunk_size": small_chunk_size,
                "large_chunk_size": large_chunk_size,
                "overlap": overlap,
                "original_length": len(text)
            }
        )
