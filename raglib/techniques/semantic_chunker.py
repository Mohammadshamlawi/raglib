"""SemanticChunker: Semantic similarity-based chunking technique."""

from typing import Any, Dict, List, Optional, Union

from ..adapters.base import EmbedderAdapter
from ..adapters.dummy_embedder import DummyEmbedder
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Document


@TechniqueRegistry.register
class SemanticChunker(RAGTechnique):
    """Semantic similarity-based chunking technique.
    
    Groups and merges adjacent text segments based on semantic similarity.
    Uses an embedder to compute segment similarities and merges similar
    adjacent segments when similarity exceeds the threshold.
    """

    meta = TechniqueMeta(
        name="Semantic Chunking",
        category="chunking",
        description="Semantic similarity-based chunking with configurable embedder"
    )

    def __init__(
        self,
        embedder: Optional[EmbedderAdapter] = None,
        chunk_size: int = 200,
        overlap: int = 50,
        similarity_threshold: float = 0.8,
        dim: int = 16
    ):
        """Initialize SemanticChunker.
        
        Args:
            embedder: Optional embedder adapter; uses DummyEmbedder if None
            chunk_size: Initial segment size before semantic merging
            overlap: Overlap between initial segments
            similarity_threshold: Similarity threshold for merging adjacent segments
            dim: Embedding dimension for fallback embedder
        """
        super().__init__(self.meta)
        self.embedder = embedder or DummyEmbedder(dim=dim)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.dim = dim

    def _compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        if len(emb1) != len(emb2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _create_initial_segments(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Create initial segments using fixed-size chunking."""
        segments = []
        step = self.chunk_size - self.overlap
        start_pos = 0
        segment_index = 0

        while start_pos < len(text):
            end_pos = min(start_pos + self.chunk_size, len(text))
            segment_text = text[start_pos:end_pos]

            segments.append({
                'text': segment_text,
                'start_idx': start_pos,
                'end_idx': end_pos,
                'index': segment_index,
                'document_id': document_id
            })

            segment_index += 1
            start_pos += step

            if end_pos == len(text):
                break

        return segments

    def _merge_segments(self, segments: List[Dict[str, Any]], embeddings: List[List[float]], full_text: str) -> List[Dict[str, Any]]:
        """Merge semantically similar adjacent segments."""
        if len(segments) <= 1:
            # Ensure single segments have merged_from field
            for seg in segments:
                if 'merged_from' not in seg:
                    seg['merged_from'] = [seg['index']]
            return segments

        merged_segments = []
        current_segment = segments[0].copy()
        current_segment['merged_from'] = [current_segment['index']]
        current_embedding = embeddings[0]

        for i in range(1, len(segments)):
            next_segment = segments[i]
            next_embedding = embeddings[i]

            # Compute similarity between current merged segment and next segment
            similarity = self._compute_similarity(current_embedding, next_embedding)

            if similarity >= self.similarity_threshold:
                # Merge segments - extend current segment to include next segment
                current_segment['end_idx'] = next_segment['end_idx']
                current_segment['text'] = full_text[current_segment['start_idx']:current_segment['end_idx']]
                current_segment['merged_from'].append(next_segment['index'])

                # Update embedding as average (simple approach)
                current_embedding = [
                    (a + b) / 2 for a, b in zip(current_embedding, next_embedding)
                ]
            else:
                # Add current segment to results and start new one
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
                current_segment['merged_from'] = [current_segment['index']]
                current_embedding = next_embedding

        # Add the last segment
        merged_segments.append(current_segment)

        return merged_segments

    def apply(self, document: Union[Document, str], *args, **kwargs) -> TechniqueResult:
        """Apply semantic chunking to document.
        
        Args:
            document: Document object or plain text string
            chunk_size: Override initial segment size (optional)
            overlap: Override initial segment overlap (optional)
            similarity_threshold: Override similarity threshold (optional)
            
        Returns:
            TechniqueResult with chunks in payload
        """
        # Extract parameters
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        overlap = kwargs.get('overlap', self.overlap)
        similarity_threshold = kwargs.get('similarity_threshold', self.similarity_threshold)

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

        # Create initial segments
        segments = self._create_initial_segments(text, document_id)

        if not segments:
            return TechniqueResult(
                success=True,
                payload={"chunks": []},
                meta={
                    "total_chunks": 0,
                    "initial_segments": 0,
                    "merged_segments": 0
                }
            )

        # Compute embeddings for all segments
        segment_texts = [seg['text'] for seg in segments]
        embeddings = self.embedder.embed(segment_texts)

        # Merge semantically similar adjacent segments
        merged_segments = self._merge_segments(segments, embeddings, text)

        # Create final chunks
        chunks = []
        for chunk_index, segment in enumerate(merged_segments):
            # Ensure merged_from field exists
            if 'merged_from' not in segment:
                segment['merged_from'] = [segment['index']]

            # Recompute embedding for merged segment if it was merged
            if len(segment['merged_from']) > 1:
                final_embedding = self.embedder.embed([segment['text']])[0]
            else:
                # Use original embedding
                original_idx = segment['merged_from'][0]
                final_embedding = embeddings[original_idx]

            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=segment['text'],
                start_idx=segment['start_idx'],
                end_idx=segment['end_idx'],
                embedding=final_embedding,
                meta={
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "similarity_threshold": similarity_threshold,
                    "chunk_index": chunk_index,
                    "merged_from": segment['merged_from'],
                    "method": "semantic_merge",
                    "embedder_type": type(self.embedder).__name__,
                    **doc_meta
                }
            )

            chunks.append(chunk)

        return TechniqueResult(
            success=True,
            payload={"chunks": chunks},
            meta={
                "total_chunks": len(chunks),
                "initial_segments": len(segments),
                "merged_segments": len(merged_segments),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "similarity_threshold": similarity_threshold,
                "original_length": len(text)
            }
        )
