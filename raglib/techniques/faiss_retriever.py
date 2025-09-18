"""FAISS (Facebook AI Similarity Search) Retriever.

A wrapper around FAISS library for efficient similarity search and clustering
of dense vectors. This implementation provides a production-ready FAISS-based
retriever with fallback to simple vector similarity when FAISS is not available.
"""
from collections.abc import Sequence
from typing import Optional

import numpy as np

from ..adapters.base import EmbedderAdapter, VectorStoreAdapter
from ..adapters.dummy_embedder import DummyEmbedder
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Hit

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSVectorStore(VectorStoreAdapter):
    """FAISS-based vector store implementation."""

    def __init__(self, dimension: int = 16, index_type: str = "IndexFlatIP"):
        """Initialize FAISS vector store.

        Args:
            dimension: Vector dimension
            index_type: FAISS index type (IndexFlatIP, IndexFlatL2, IndexIVFFlat, etc.)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metadata_store = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0

        if FAISS_AVAILABLE:
            if index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(dimension)
            elif index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(dimension)
            elif index_type == "IndexIVFFlat":
                # For IVF, we need a quantizer
                quantizer = faiss.IndexFlatL2(dimension)
                n_centroids = min(100, max(1, dimension // 2))
                self.index = faiss.IndexIVFFlat(quantizer, dimension, n_centroids)
            else:
                # Default to flat IP
                self.index = faiss.IndexFlatIP(dimension)
        else:
            # Fallback implementation without FAISS
            self.vectors = []

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadata: Optional[Sequence[dict]] = None
    ) -> None:
        """Add vectors to FAISS index."""
        vectors_array = np.array(vectors, dtype=np.float32)

        if FAISS_AVAILABLE:
            # Train index if needed (for IVF indices)
            if not self.index.is_trained:
                self.index.train(vectors_array)

            self.index.add(vectors_array)
        else:
            # Fallback storage
            self.vectors.extend(vectors_array.tolist())

        # Store metadata and ID mapping
        for i, id_val in enumerate(ids):
            idx = self.next_idx + i
            self.id_to_idx[id_val] = idx
            self.idx_to_id[idx] = id_val
            if metadata:
                self.metadata_store[id_val] = metadata[i]

        self.next_idx += len(ids)

    def search(
        self, query_vector: Sequence[float], top_k: int = 5
    ) -> list[tuple]:
        """Search for similar vectors."""
        query_array = np.array([query_vector], dtype=np.float32)

        if FAISS_AVAILABLE and hasattr(self, 'index') and self.index.ntotal > 0:
            search_k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_array, search_k)

            results = []
            for _, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    doc_id = self.idx_to_id.get(idx, str(idx))
                    metadata = self.metadata_store.get(doc_id, {})
                    results.append((doc_id, float(score), metadata))

            return results
        else:
            # Fallback implementation
            if not hasattr(self, 'vectors') or not self.vectors:
                return []

            similarities = []
            query_norm = np.linalg.norm(query_vector)

            for i, vec in enumerate(self.vectors):
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 0 and query_norm > 0:
                    similarity = np.dot(query_vector, vec) / (query_norm * vec_norm)
                else:
                    similarity = 0.0
                similarities.append((i, similarity))

            # Sort by similarity descending
            similarities.sort(key=lambda x: x[1], reverse=True)

            results = []
            for _, (idx, score) in enumerate(similarities[:top_k]):
                doc_id = self.idx_to_id.get(idx, str(idx))
                metadata = self.metadata_store.get(doc_id, {})
                results.append((doc_id, score, metadata))

            return results


@TechniqueRegistry.register
class FAISSRetriever(RAGTechnique):
    """FAISS-based dense retriever for efficient similarity search.
    
    Uses FAISS library for fast approximate nearest neighbor search when available,
    falls back to simple cosine similarity otherwise.
    """
    
    meta = TechniqueMeta(
        name="faiss_retriever",
        category="core_retrieval",
        description="FAISS-based dense retriever with efficient similarity search",
        tags={
            "type": "dense_retrieval",
            "library": "faiss",
            "fallback": "cosine_similarity"
        }
    )
    
    def __init__(
        self,
        embedder: Optional[EmbedderAdapter] = None,
        dimension: int = 16,
        index_type: str = "IndexFlatIP",
        chunks: Optional[Sequence[Chunk]] = None
    ):
        """Initialize FAISS retriever.
        
        Args:
            embedder: Embedder adapter for generating embeddings
            dimension: Vector dimension for FAISS index
            index_type: FAISS index type (IndexFlatIP, IndexFlatL2, IndexIVFFlat)
            chunks: Optional initial chunks to add
        """
        super().__init__(self.meta)
        
        self.embedder = (
            embedder if embedder is not None else DummyEmbedder(dim=dimension)
        )
        self.vectorstore = FAISSVectorStore(
            dimension=dimension, index_type=index_type
        )
        self.dimension = dimension
        
        if chunks:
            self.add_chunks(chunks)
    
    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the FAISS index."""
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
                "original_text": chunk.text
            })
        
        # Generate embeddings
        vectors = self.embedder.embed(texts)
        
        # Add to FAISS index
        self.vectorstore.add(ids, vectors, metas)
    
    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply FAISS retrieval.
        
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
                        "method": "faiss_add_chunks",
                        "library_available": FAISS_AVAILABLE
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
                    "method": "faiss_search",
                    "library_available": FAISS_AVAILABLE
                }
            )
        
        # Generate query embedding
        try:
            query_vectors = self.embedder.embed([query])
            if not query_vectors:
                return TechniqueResult(
                    success=False,
                    payload={"error": "Failed to generate query embedding"},
                    meta={
                        "method": "faiss_search",
                        "library_available": FAISS_AVAILABLE
                    }
                )
            
            query_vector = query_vectors[0]
            
            # Search using FAISS
            search_results = self.vectorstore.search(query_vector, top_k=top_k)
            
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
                    "method": "faiss_search",
                    "library_available": FAISS_AVAILABLE,
                    "query_length": len(query),
                    "results_count": len(hits)
                }
            )

        except Exception as e:
            return TechniqueResult(
                success=False,
                payload={"error": f"FAISS search failed: {str(e)}"},
                meta={"method": "faiss_search", "library_available": FAISS_AVAILABLE}
            )