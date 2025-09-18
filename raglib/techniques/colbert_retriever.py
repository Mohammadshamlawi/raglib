"""ColBERT-style Late Interaction Model.

Implements a simplified version of ColBERT's late interaction mechanism
where each token in the query and document gets its own embedding,
and similarity is computed via maximum similarity aggregation.
"""
from collections.abc import Sequence
from typing import Optional

import numpy as np

from ..adapters.base import EmbedderAdapter
from ..adapters.dummy_embedder import DummyEmbedder
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Document, Hit


def _simple_tokenize(text: str) -> list[str]:
    """Simple tokenization by splitting on whitespace."""
    return text.lower().split()


def _pad_embeddings(
    embeddings: list[list[float]], max_length: int
) -> list[list[float]]:
    """Pad embeddings to max_length with zero vectors."""
    if not embeddings:
        return []

    dim = len(embeddings[0])
    padded = embeddings.copy()

    while len(padded) < max_length:
        padded.append([0.0] * dim)

    return padded[:max_length]


def _colbert_similarity(
    query_embeddings: list[list[float]],
    doc_embeddings: list[list[float]]
) -> float:
    """Compute ColBERT-style late interaction similarity.

    For each query token embedding, find the most similar document token
    embedding and sum these maximum similarities.
    """
    if not query_embeddings or not doc_embeddings:
        return 0.0

    total_similarity = 0.0
    query_array = np.array(query_embeddings)
    doc_array = np.array(doc_embeddings)

    # For each query token, find max similarity with any doc token
    for q_emb in query_array:
        if np.linalg.norm(q_emb) == 0:  # Skip zero embeddings (padding)
            continue

        similarities = []
        for d_emb in doc_array:
            if np.linalg.norm(d_emb) == 0:  # Skip zero embeddings (padding)
                continue

            # Cosine similarity
            q_norm = np.linalg.norm(q_emb)
            d_norm = np.linalg.norm(d_emb)
            if q_norm > 0 and d_norm > 0:
                sim = np.dot(q_emb, d_emb) / (q_norm * d_norm)
                similarities.append(sim)

        if similarities:
            total_similarity += max(similarities)

    return total_similarity


@TechniqueRegistry.register
class ColBERTRetriever(RAGTechnique):
    """ColBERT-style late interaction retriever.

    Computes token-level embeddings and uses late interaction
    (maximum similarity aggregation) for retrieval scoring.
    """

    meta = TechniqueMeta(
        name="colbert_retriever",
        category="core_retrieval",
        description="ColBERT-style late interaction dense retrieval",
        tags={
            "type": "dense_retrieval",
            "architecture": "late_interaction",
            "token_level": "true"
        }
    )

    def __init__(
        self,
        embedder: Optional[EmbedderAdapter] = None,
        max_query_length: int = 32,
        max_doc_length: int = 180,
        dim: int = 16
    ):
        """Initialize ColBERT retriever.

        Args:
            embedder: Token-level embedder (if None, uses dummy embedder)
            max_query_length: Maximum number of query tokens
            max_doc_length: Maximum number of document tokens
            dim: Embedding dimension
        """
        super().__init__(self.meta)

        self.embedder = (
            embedder if embedder is not None
            else DummyEmbedder(dim=dim)
        )
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.dim = dim

        # Store document embeddings for retrieval
        self.doc_embeddings = {}  # doc_id -> token embeddings
        self.doc_metadata = {}    # doc_id -> metadata

    def _encode_tokens(self, text: str, max_length: int) -> list[list[float]]:
        """Encode text tokens into embeddings."""
        tokens = _simple_tokenize(text)

        if not tokens:
            return [[0.0] * self.dim] * max_length

        # Truncate to max length
        tokens = tokens[:max_length]

        # Get embeddings for tokens
        embeddings = self.embedder.embed(tokens)

        # Pad to max length
        padded_embeddings = _pad_embeddings(embeddings, max_length)

        return padded_embeddings

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks by encoding their tokens."""
        for chunk in chunks:
            # Encode document tokens
            doc_embeddings = self._encode_tokens(chunk.text, self.max_doc_length)

            self.doc_embeddings[chunk.id] = doc_embeddings
            self.doc_metadata[chunk.id] = {
                "chunk": chunk,
                "document_id": chunk.document_id,
                "original_text": chunk.text
            }

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Add documents by encoding their tokens."""
        for doc in documents:
            # Encode document tokens
            doc_embeddings = self._encode_tokens(doc.text, self.max_doc_length)

            self.doc_embeddings[doc.id] = doc_embeddings
            self.doc_metadata[doc.id] = {
                "document": doc,
                "document_id": doc.id,
                "original_text": doc.text
            }

    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply ColBERT late interaction retrieval.

        Usage:
            # Add chunks
            result = retriever.apply(chunks=[...])

            # Add documents
            result = retriever.apply(documents=[...])

            # Query
            result = retriever.apply(query="search query", top_k=10)
        """
        # Handle chunk addition
        if args:
            first = args[0]
            if isinstance(first, (list, tuple)) and first:
                if hasattr(first[0], "text"):
                    if hasattr(first[0], "document_id"):  # Chunks
                        self.add_chunks(first)
                        return TechniqueResult(
                            success=True,
                            payload={"added_chunks": len(first)},
                            meta={
                                "method": "colbert_add_chunks",
                                "max_doc_length": self.max_doc_length
                            }
                        )
                    else:  # Documents
                        self.add_documents(first)
                        return TechniqueResult(
                            success=True,
                            payload={"added_documents": len(first)},
                            meta={
                                "method": "colbert_add_documents",
                                "max_doc_length": self.max_doc_length
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
                    "method": "colbert_search",
                    "max_query_length": self.max_query_length
                }
            )

        if not self.doc_embeddings:
            return TechniqueResult(
                success=True,
                payload={"hits": []},
                meta={
                    "method": "colbert_search",
                    "message": "No documents indexed"
                }
            )

        try:
            # Encode query tokens
            query_embeddings = self._encode_tokens(query, self.max_query_length)

            # Compute similarities with all documents
            similarities = []
            for doc_id, doc_embeddings in self.doc_embeddings.items():
                similarity = _colbert_similarity(query_embeddings, doc_embeddings)
                similarities.append((doc_id, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Create hits
            hits: list[Hit] = []
            for doc_id, score in similarities[:top_k]:
                metadata = self.doc_metadata.get(doc_id, {})
                chunk_obj = metadata.get("chunk")
                document_id = metadata.get("document_id", doc_id)

                hits.append(Hit(
                    doc_id=document_id,
                    score=float(score),
                    chunk=chunk_obj,
                    meta=metadata
                ))

            return TechniqueResult(
                success=True,
                payload={"hits": hits},
                meta={
                    "method": "colbert_search",
                    "max_query_length": self.max_query_length,
                    "max_doc_length": self.max_doc_length,
                    "query_length": len(_simple_tokenize(query)),
                    "results_count": len(hits),
                    "total_docs": len(self.doc_embeddings)
                }
            )

        except Exception as e:
            return TechniqueResult(
                success=False,
                payload={"error": f"ColBERT search failed: {str(e)}"},
                meta={
                    "method": "colbert_search",
                    "max_query_length": self.max_query_length,
                    "max_doc_length": self.max_doc_length
                }
            )
