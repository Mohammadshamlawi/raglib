"""Multi-Vector Retrieval.

Implements multi-vector retrieval where each document is represented by
multiple embeddings (e.g., from different parts or views of the document),
allowing for more nuanced similarity computation.
"""
from collections.abc import Sequence
from typing import Optional

import numpy as np

from ..adapters.base import EmbedderAdapter
from ..adapters.dummy_embedder import DummyEmbedder
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Hit


def _split_into_segments(text: str, max_segment_length: int = 100) -> list[str]:
    """Split text into segments for multi-vector representation."""
    words = text.split()
    segments = []

    current_segment = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_segment_length and current_segment:
            segments.append(" ".join(current_segment))
            current_segment = [word]
            current_length = len(word)
        else:
            current_segment.append(word)
            current_length += len(word) + 1

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments if segments else [text]


def _max_sim_aggregation(
    query_vector: list[float], doc_vectors: list[list[float]]
) -> float:
    """Compute maximum similarity between query and any document vector."""
    if not doc_vectors:
        return 0.0

    query_array = np.array(query_vector)
    query_norm = np.linalg.norm(query_array)

    if query_norm == 0:
        return 0.0

    max_similarity = -float('inf')

    for doc_vector in doc_vectors:
        doc_array = np.array(doc_vector)
        doc_norm = np.linalg.norm(doc_array)

        if doc_norm == 0:
            continue

        # Cosine similarity
        similarity = np.dot(query_array, doc_array) / (query_norm * doc_norm)
        max_similarity = max(max_similarity, similarity)

    return float(max_similarity) if max_similarity != -float('inf') else 0.0


def _avg_sim_aggregation(
    query_vector: list[float], doc_vectors: list[list[float]]
) -> float:
    """Compute average similarity between query and document vectors."""
    if not doc_vectors:
        return 0.0

    query_array = np.array(query_vector)
    query_norm = np.linalg.norm(query_array)

    if query_norm == 0:
        return 0.0

    similarities = []

    for doc_vector in doc_vectors:
        doc_array = np.array(doc_vector)
        doc_norm = np.linalg.norm(doc_array)

        if doc_norm == 0:
            continue

        # Cosine similarity
        similarity = np.dot(query_array, doc_array) / (query_norm * doc_norm)
        similarities.append(similarity)

    return float(np.mean(similarities)) if similarities else 0.0


@TechniqueRegistry.register
class MultiVectorRetriever(RAGTechnique):
    """Multi-Vector Retrieval technique.

    Represents each document with multiple embeddings and uses
    aggregation strategies for similarity computation.
    """

    meta = TechniqueMeta(
        name="multi_vector_retriever",
        category="core_retrieval",
        description="Multi-vector dense retrieval with document segmentation",
        tags={
            "type": "dense_retrieval",
            "multi_vector": "true",
            "aggregation": "max_avg"
        }
    )

    def __init__(
        self,
        embedder: Optional[EmbedderAdapter] = None,
        max_segment_length: int = 100,
        aggregation_method: str = "max",
        chunks: Optional[Sequence[Chunk]] = None,
        dim: int = 16
    ):
        """Initialize Multi-Vector Retriever.

        Args:
            embedder: Embedder for generating embeddings
            max_segment_length: Maximum length for document segments
            aggregation_method: Aggregation method ("max", "avg")
            chunks: Optional initial chunks
            dim: Embedding dimension for dummy embedder
        """
        super().__init__(self.meta)

        self.embedder = (
            embedder if embedder is not None
            else DummyEmbedder(dim=dim)
        )
        self.max_segment_length = max_segment_length
        self.aggregation_method = aggregation_method

        # Store multi-vector representations
        self.doc_vectors = {}  # doc_id -> list of vectors
        self.doc_metadata = {}  # doc_id -> metadata

        if chunks:
            self.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks by creating multi-vector representations."""
        for chunk in chunks:
            # Split text into segments
            segments = _split_into_segments(chunk.text, self.max_segment_length)

            # Embed each segment
            if segments:
                segment_vectors = self.embedder.embed(segments)
                self.doc_vectors[chunk.id] = segment_vectors
            else:
                # Fallback: embed entire text
                full_vectors = self.embedder.embed([chunk.text])
                self.doc_vectors[chunk.id] = full_vectors

            # Store metadata
            chunk_meta = {
                "chunk": chunk,
                "document_id": chunk.document_id,
                "num_segments": len(segments),
                "segments": segments
            }
            # Add original chunk metadata if it exists
            if hasattr(chunk, 'metadata') and chunk.metadata:
                chunk_meta.update(chunk.metadata)
            self.doc_metadata[chunk.id] = chunk_meta

    def _compute_document_similarity(
        self, query_vector: list[float], doc_id: str
    ) -> float:
        """Compute similarity between query and a multi-vector document."""
        doc_vectors = self.doc_vectors.get(doc_id, [])

        if not doc_vectors:
            return 0.0

        if self.aggregation_method == "max":
            return _max_sim_aggregation(query_vector, doc_vectors)
        elif self.aggregation_method == "avg":
            return _avg_sim_aggregation(query_vector, doc_vectors)
        else:
            # Default to max
            return _max_sim_aggregation(query_vector, doc_vectors)

    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply multi-vector retrieval.

        Usage:
            # Add chunks
            result = retriever.apply(chunks=[...])

            # Multi-vector search
            result = retriever.apply(query="search query", top_k=10)
        """
        # Handle chunk addition
        if args:
            first = args[0]
            if isinstance(first, (list, tuple)) and first and hasattr(first[0], "text"):
                self.add_chunks(first)
                return TechniqueResult(
                    success=True,
                    payload={"added": len(first)},
                    meta={
                        "method": "multi_vector_add_chunks",
                        "max_segment_length": self.max_segment_length,
                        "aggregation_method": self.aggregation_method
                    }
                )

        # Handle query
        query = kwargs.get("query", "") or (args[1] if len(args) > 1 else "")
        top_k = kwargs.get("top_k", 5)

        if not query:
            return TechniqueResult(
                success=True,
                payload={"hits": []},
                meta={
                    "method": "multi_vector_search",
                    "aggregation_method": self.aggregation_method
                }
            )

        if not self.doc_vectors:
            return TechniqueResult(
                success=True,
                payload={"hits": []},
                meta={
                    "method": "multi_vector_search",
                    "message": "No documents indexed"
                }
            )

        try:
            # Embed query
            query_vectors = self.embedder.embed([query])
            if not query_vectors:
                return TechniqueResult(
                    success=False,
                    payload={"error": "Failed to embed query"},
                    meta={
                        "method": "multi_vector_search",
                        "aggregation_method": self.aggregation_method
                    }
                )

            query_vector = query_vectors[0]

            # Compute similarities with all documents
            similarities = []
            for doc_id in self.doc_vectors:
                similarity = self._compute_document_similarity(query_vector, doc_id)
                similarities.append((doc_id, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Create hits
            hits: list[Hit] = []
            for doc_id, score in similarities[:top_k]:
                metadata = self.doc_metadata.get(doc_id, {})
                chunk_obj = metadata.get("chunk")
                document_id = metadata.get("document_id", doc_id)

                # Add multi-vector specific metadata
                enhanced_meta = dict(metadata)
                enhanced_meta.update({
                    "aggregation_method": self.aggregation_method,
                    "multi_vector_score": score
                })

                hits.append(Hit(
                    doc_id=document_id,
                    score=float(score),
                    chunk=chunk_obj,
                    meta=enhanced_meta
                ))

            return TechniqueResult(
                success=True,
                payload={"hits": hits},
                meta={
                    "method": "multi_vector_search",
                    "aggregation_method": self.aggregation_method,
                    "max_segment_length": self.max_segment_length,
                    "query_length": len(query),
                    "results_count": len(hits),
                    "total_docs": len(self.doc_vectors)
                }
            )

        except Exception as e:
            return TechniqueResult(
                success=False,
                payload={"error": f"Multi-vector search failed: {str(e)}"},
                meta={
                    "method": "multi_vector_search",
                    "aggregation_method": self.aggregation_method
                }
            )
