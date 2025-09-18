"""Multi-Query Retrieval.

Generates multiple query variations and retrieves results for each,
then combines the results using various fusion strategies.
"""
from collections import defaultdict
from collections.abc import Sequence
from typing import Optional

from ..adapters.base import EmbedderAdapter, LLMAdapter, VectorStoreAdapter
from ..adapters.dummy_embedder import DummyEmbedder
from ..adapters.inmemory_vectorstore import InMemoryVectorStore
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Hit


def _reciprocal_rank_fusion(
    hit_lists: list[list[Hit]], k: int = 60
) -> list[Hit]:
    """Combine multiple hit lists using Reciprocal Rank Fusion (RRF)."""
    doc_scores = defaultdict(float)
    doc_hits = {}

    for hit_list in hit_lists:
        for rank, hit in enumerate(hit_list):
            # RRF score: 1 / (k + rank)
            score = 1.0 / (k + rank + 1)  # +1 for 0-based indexing
            doc_scores[hit.doc_id] += score
            doc_hits[hit.doc_id] = hit  # Keep one instance

    # Sort by combined score
    sorted_docs = sorted(
        doc_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Create final hit list with RRF scores
    combined_hits = []
    for doc_id, rrf_score in sorted_docs:
        hit = doc_hits[doc_id]
        # Update hit with RRF score
        combined_hit = Hit(
            doc_id=hit.doc_id,
            score=rrf_score,
            chunk=hit.chunk,
            meta={**hit.meta, "rrf_score": rrf_score, "fusion_method": "rrf"}
        )
        combined_hits.append(combined_hit)

    return combined_hits


def _generate_query_variations(
    original_query: str,
    llm_adapter: Optional[LLMAdapter] = None,
    num_variations: int = 3
) -> list[str]:
    """Generate query variations."""
    variations = [original_query]  # Always include original

    if llm_adapter:
        try:
            num_variations_needed = num_variations - 1
            prompt = f"""Generate {num_variations_needed} alternative phrasings:
"{original_query}"

Requirements:
- Keep the same meaning and intent
- Use different wording and structure
- Make them suitable for search/retrieval
- Output one variation per line

Variations:"""

            response = llm_adapter.generate(prompt)
            lines = response.strip().split('\n')
            for line in lines[:num_variations-1]:
                line = line.strip()
                if line and line != original_query:
                    variations.append(line)

        except Exception:
            # Fallback to simple variations
            pass

    # Fallback: simple variations if we don't have enough
    while len(variations) < num_variations:
        if len(variations) == 1:
            variations.append(f"What is {original_query}?")
        elif len(variations) == 2:
            variations.append(f"Tell me about {original_query}")
        else:
            break

    return variations[:num_variations]


@TechniqueRegistry.register
class MultiQueryRetriever(RAGTechnique):
    """Multi-Query Retrieval technique.

    Generates multiple query variations, retrieves results for each,
    and combines them using fusion strategies like Reciprocal Rank Fusion.
    """

    meta = TechniqueMeta(
        name="multi_query_retriever",
        category="retrieval_enhancement",
        description="Multi-query retrieval with result fusion",
        tags={
            "type": "query_expansion",
            "fusion": "reciprocal_rank_fusion",
            "multi_query": "true"
        }
    )

    def __init__(
        self,
        embedder: Optional[EmbedderAdapter] = None,
        vectorstore: Optional[VectorStoreAdapter] = None,
        llm_adapter: Optional[LLMAdapter] = None,
        num_queries: int = 3,
        fusion_method: str = "rrf",
        fusion_k: int = 60,
        chunks: Optional[Sequence[Chunk]] = None,
        dim: int = 16
    ):
        """Initialize Multi-Query Retriever.

        Args:
            embedder: Embedder for generating embeddings
            vectorstore: Vector store for retrieval
            llm_adapter: LLM for generating query variations
            num_queries: Number of query variations to generate
            fusion_method: Fusion method ("rrf", "simple_avg")
            fusion_k: RRF parameter k
            chunks: Optional initial chunks
            dim: Embedding dimension for dummy embedder
        """
        super().__init__(self.meta)

        self.embedder = (
            embedder if embedder is not None
            else DummyEmbedder(dim=dim)
        )
        self.vectorstore = (
            vectorstore if vectorstore is not None
            else InMemoryVectorStore()
        )
        self.llm_adapter = llm_adapter
        self.num_queries = num_queries
        self.fusion_method = fusion_method
        self.fusion_k = fusion_k

        if chunks:
            self.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the vector store."""
        if not chunks:
            return

        ids = []
        texts = []
        metas = []

        for chunk in chunks:
            ids.append(chunk.id)
            texts.append(chunk.text)
            # Merge chunk metadata with our required metadata
            chunk_meta = {
                "chunk": chunk,
                "document_id": chunk.document_id
            }
            # Add original chunk metadata if it exists
            if hasattr(chunk, 'metadata') and chunk.metadata:
                chunk_meta.update(chunk.metadata)
            metas.append(chunk_meta)

        vectors = self.embedder.embed(texts)
        self.vectorstore.add(ids, vectors, metas)

    def _retrieve_for_query(
        self, query: str, top_k: int
    ) -> list[Hit]:
        """Retrieve results for a single query."""
        try:
            query_vectors = self.embedder.embed([query])
            if not query_vectors:
                return []

            query_vector = query_vectors[0]
            search_results = self.vectorstore.search(query_vector, top_k=top_k)

            hits = []
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

            return hits

        except Exception:
            return []

    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply multi-query retrieval.

        Usage:
            # Add chunks
            result = retriever.apply(chunks=[...])

            # Multi-query search
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
                        "method": "multi_query_add_chunks",
                        "num_queries": self.num_queries
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
                    "method": "multi_query_search",
                    "num_queries": self.num_queries
                }
            )

        try:
            # Generate query variations
            query_variations = _generate_query_variations(
                query, self.llm_adapter, self.num_queries
            )

            # Retrieve for each query variation
            all_hits = []
            for i, variation in enumerate(query_variations):
                hits = self._retrieve_for_query(variation, top_k * 2)
                # Add query info to metadata
                for hit in hits:
                    hit.meta["query_variation"] = variation
                    hit.meta["query_index"] = i
                all_hits.append(hits)

            # Fuse results
            if self.fusion_method == "rrf":
                final_hits = _reciprocal_rank_fusion(all_hits, k=self.fusion_k)
            else:
                # Simple concatenation fallback
                final_hits = []
                for hits in all_hits:
                    final_hits.extend(hits)

            # Limit to top_k
            final_hits = final_hits[:top_k]

            return TechniqueResult(
                success=True,
                payload={"hits": final_hits},
                meta={
                    "method": "multi_query_search",
                    "num_queries": self.num_queries,
                    "fusion_method": self.fusion_method,
                    "query_variations": query_variations,
                    "results_count": len(final_hits)
                }
            )

        except Exception as e:
            return TechniqueResult(
                success=False,
                payload={"error": f"Multi-query search failed: {str(e)}"},
                meta={
                    "method": "multi_query_search",
                    "num_queries": self.num_queries,
                    "fusion_method": self.fusion_method
                }
            )
