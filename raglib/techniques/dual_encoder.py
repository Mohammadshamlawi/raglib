"""Dual-Encoder Dense Retrieval.

Implements dual-encoder architecture where query and documents are encoded
separately by two neural networks (or the same network applied separately).
This is the foundation for many modern dense retrieval systems.
"""
from collections.abc import Sequence
from typing import Optional

import numpy as np

from ..adapters.base import EmbedderAdapter, VectorStoreAdapter
from ..adapters.dummy_embedder import DummyEmbedder
from ..adapters.inmemory_vectorstore import InMemoryVectorStore
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Hit


@TechniqueRegistry.register
class DualEncoder(RAGTechnique):
    """Dual-Encoder dense retrieval technique.

    Encodes queries and documents separately using (potentially different)
    encoders, then computes similarity in the shared embedding space.
    """

    meta = TechniqueMeta(
        name="dual_encoder",
        category="core_retrieval",
        description="Dual-encoder dense retrieval with separate query/doc encoding",
        tags={
            "type": "dense_retrieval",
            "architecture": "dual_encoder",
            "supports": "asymmetric_encoding"
        }
    )

    def __init__(
        self,
        query_encoder: Optional[EmbedderAdapter] = None,
        doc_encoder: Optional[EmbedderAdapter] = None,
        vectorstore: Optional[VectorStoreAdapter] = None,
        chunks: Optional[Sequence[Chunk]] = None,
        dim: int = 16,
        similarity_metric: str = "cosine"
    ):
        """Initialize Dual-Encoder retriever.

        Args:
            query_encoder: Encoder for queries (if None, uses dummy encoder)
            doc_encoder: Encoder for documents (if None, uses query_encoder or dummy)
            vectorstore: Vector store (if None, uses in-memory store)
            chunks: Optional initial chunks to encode and store
            dim: Embedding dimension for dummy encoders
            similarity_metric: Similarity metric (cosine, dot_product, l2)
        """
        super().__init__(self.meta)

        # Set up encoders
        self.query_encoder = (
            query_encoder if query_encoder is not None
            else DummyEmbedder(dim=dim)
        )
        self.doc_encoder = (
            doc_encoder if doc_encoder is not None
            else self.query_encoder  # Share encoder if not specified
        )

        # Set up vector store
        self.vectorstore = (
            vectorstore if vectorstore is not None
            else InMemoryVectorStore()
        )

        self.similarity_metric = similarity_metric
        self.dim = dim

        # Add initial chunks if provided
        if chunks:
            self.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks using document encoder."""
        if not chunks:
            return

        ids = []
        texts = []
        metas = []

        for chunk in chunks:
            ids.append(chunk.id)
            texts.append(chunk.text)
            metas.append({
                "chunk": chunk,
                "document_id": chunk.document_id,
                "encoder": "doc_encoder"
            })

        # Encode documents using document encoder
        vectors = self.doc_encoder.embed(texts)
        self.vectorstore.add(ids, vectors, metas)

    def _compute_similarity(
        self, query_vec: list[float], doc_vecs: list[list[float]]
    ) -> list[float]:
        """Compute similarity scores between query and document vectors."""
        query_array = np.array(query_vec)
        doc_arrays = np.array(doc_vecs)

        if self.similarity_metric == "cosine":
            # Cosine similarity
            query_norm = np.linalg.norm(query_array)
            doc_norms = np.linalg.norm(doc_arrays, axis=1)

            if query_norm == 0:
                return [0.0] * len(doc_vecs)

            similarities = []
            for i, doc_vec in enumerate(doc_arrays):
                if doc_norms[i] == 0:
                    similarities.append(0.0)
                else:
                    sim = np.dot(query_array, doc_vec) / (query_norm * doc_norms[i])
                    similarities.append(float(sim))
            return similarities

        elif self.similarity_metric == "dot_product":
            # Dot product similarity
            similarities = []
            for doc_vec in doc_arrays:
                sim = np.dot(query_array, doc_vec)
                similarities.append(float(sim))
            return similarities

        elif self.similarity_metric == "l2":
            # Negative L2 distance (higher is more similar)
            similarities = []
            for doc_vec in doc_arrays:
                dist = np.linalg.norm(query_array - doc_vec)
                similarities.append(-float(dist))
            return similarities

        else:
            # Default to cosine
            return self._compute_similarity(query_vec, doc_vecs)

    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply dual-encoder retrieval.

        Usage:
            # Add chunks
            result = retriever.apply(chunks=[...])

            # Query
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
                        "method": "dual_encoder_add_chunks",
                        "doc_encoder": str(type(self.doc_encoder).__name__)
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
                    "method": "dual_encoder_search",
                    "query_encoder": str(type(self.query_encoder).__name__)
                }
            )

        try:
            # Encode query using query encoder
            query_vectors = self.query_encoder.embed([query])
            if not query_vectors:
                return TechniqueResult(
                    success=False,
                    payload={"error": "Failed to encode query"},
                    meta={
                        "method": "dual_encoder_search",
                        "query_encoder": str(type(self.query_encoder).__name__)
                    }
                )

            query_vector = query_vectors[0]

            # For dual-encoder, we need to retrieve and re-rank
            # if using custom similarity
            if hasattr(self.vectorstore, "search"):
                # Use vector store's built-in search
                search_results = self.vectorstore.search(query_vector, top_k=top_k)
            else:
                # Fallback: manual search (not efficient for large collections)
                search_results = []

            # Convert to Hit objects
            hits: list[Hit] = []
            for doc_id, score, metadata in search_results:
                chunk_obj = metadata.get("chunk") if metadata else None
                document_id = (
                    metadata.get("document_id", doc_id) if metadata else doc_id
                )

                hits.append(Hit(
                    doc_id=document_id,
                    score=float(score),
                    chunk=chunk_obj,
                    meta=metadata or {}
                ))

            return TechniqueResult(
                success=True,
                payload={"hits": hits},
                meta={
                    "method": "dual_encoder_search",
                    "query_encoder": str(type(self.query_encoder).__name__),
                    "doc_encoder": str(type(self.doc_encoder).__name__),
                    "similarity_metric": self.similarity_metric,
                    "query_length": len(query),
                    "results_count": len(hits)
                }
            )

        except Exception as e:
            return TechniqueResult(
                success=False,
                payload={"error": f"Dual-encoder search failed: {str(e)}"},
                meta={
                    "method": "dual_encoder_search",
                    "query_encoder": str(type(self.query_encoder).__name__),
                    "doc_encoder": str(type(self.doc_encoder).__name__)
                }
            )
