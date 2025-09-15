"""MMR (Maximal Marginal Relevance) re-ranking technique.

This module implements MMR re-ranking to balance relevance and diversity 
in search results. Supports both adapter-backed and adapterless modes.
"""

import math
from collections.abc import Sequence
from typing import Any, Callable, List, Optional

from ..adapters.dummy_embedder import DummyEmbedder
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Hit


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


@TechniqueRegistry.register
class MMRReRanker(RAGTechnique):
    """MMR (Maximal Marginal Relevance) re-ranker.
    
    Balances relevance to query and diversity among selected documents.
    
    Args:
        embedder: Optional embedder adapter for computing embeddings
        similarity_fn: Optional custom similarity function
        lambda_param: Trade-off between relevance and diversity (0.0-1.0)
        dim: Embedding dimension for fallback embedder
    """

    meta = TechniqueMeta(
        name="Maximal Marginal Relevance (MMR)",
        category="reranking",
        description="Maximal Marginal Relevance re-ranking for balancing relevance and diversity"
    )

    def __init__(self,
                 embedder: Optional[Any] = None,
                 similarity_fn: Optional[Callable[[Sequence[float], Sequence[float]], float]] = None,
                 lambda_param: float = 0.7,
                 dim: int = 16):
        super().__init__(self.meta)
        self.embedder = embedder
        self.similarity_fn = similarity_fn or _cosine_similarity
        self.lambda_param = lambda_param
        self.dim = dim
        self._fallback_embedder = None

    def _get_embedder(self):
        """Get embedder (either provided or fallback)."""
        if self.embedder is not None:
            return self.embedder
        if self._fallback_embedder is None:
            self._fallback_embedder = DummyEmbedder(dim=self.dim)
        return self._fallback_embedder

    def _get_query_embedding(self, query: Optional[str], query_embedding: Optional[List[float]]) -> List[float]:
        """Get or compute query embedding."""
        if query_embedding is not None:
            return query_embedding

        if query is None:
            return [0.0] * self.dim

        embedder = self._get_embedder()
        embeddings = embedder.embed([query])
        return embeddings[0]

    def _get_hit_embedding(self, hit: Hit) -> List[float]:
        """Get or compute embedding for a hit."""
        # Try to get embedding from chunk
        if hit.chunk and hit.chunk.embedding:
            return hit.chunk.embedding

        # Try to get embedding from hit metadata
        if "embedding" in hit.meta:
            return hit.meta["embedding"]

        # Compute embedding from text
        text = ""
        if hit.chunk and hit.chunk.text:
            text = hit.chunk.text
        elif "text" in hit.meta:
            text = hit.meta["text"]

        embedder = self._get_embedder()
        embeddings = embedder.embed([text])
        return embeddings[0]

    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply MMR re-ranking.
        
        Expected inputs:
            hits: List[Hit] - candidate documents to re-rank
            query: Optional[str] - query string  
            query_embedding: Optional[List[float]] - precomputed query embedding
            top_k: Optional[int] - number of results to return (default: 5)
        """
        # Extract parameters
        hits = None
        query = None
        query_embedding = None
        top_k = 5

        # Handle positional args
        if len(args) >= 1:
            hits = args[0]
        if len(args) >= 2:
            query = args[1]
        if len(args) >= 3:
            query_embedding = args[2]
        if len(args) >= 4:
            top_k = args[3]

        # Handle keyword args
        hits = kwargs.get('hits', hits)
        query = kwargs.get('query', query)
        query_embedding = kwargs.get('query_embedding', query_embedding)
        top_k = kwargs.get('top_k', top_k)

        # Validate inputs
        if not hits:
            return TechniqueResult(success=True, payload={"hits": []})

        # Get query embedding
        query_emb = self._get_query_embedding(query, query_embedding)

        # Get document embeddings
        doc_embeddings = []
        for hit in hits:
            doc_emb = self._get_hit_embedding(hit)
            doc_embeddings.append(doc_emb)

        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(hits)))

        while len(selected_indices) < top_k and remaining_indices:
            best_idx = None
            best_score = float('-inf')

            for idx in remaining_indices:
                doc_emb = doc_embeddings[idx]

                # Relevance to query
                relevance = self.similarity_fn(query_emb, doc_emb)

                # Maximum similarity to already selected documents
                max_sim = 0.0
                if selected_indices:
                    similarities = [
                        self.similarity_fn(doc_emb, doc_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    ]
                    max_sim = max(similarities)

                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim

                # Tie-breaking: prefer earlier index for determinism
                if mmr_score > best_score or (mmr_score == best_score and (best_idx is None or idx < best_idx)):
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        # Return selected hits in MMR order
        selected_hits = [hits[i] for i in selected_indices]

        return TechniqueResult(
            success=True,
            payload={"hits": selected_hits},
            meta={"algorithm": "mmr", "lambda": self.lambda_param, "selected_count": len(selected_hits)}
        )
