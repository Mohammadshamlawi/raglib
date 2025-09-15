"""Production-friendly Dense Retriever.

This class tries to be "production-ready" while preserving the abstract apply() contract.
- If the user supplies adapters (embedder: EmbedderAdapter, vectorstore: VectorStoreAdapter) at construction,
  those will be used.
- If adapters are not supplied, DenseRetriever will fall back to default internal adapters:
  DummyEmbedder (deterministic sha256) and InMemoryVectorStore.
- The class works without requiring adapters: the user can call apply() with chunks added via
  add_chunks(...) or passed in the apply() call.

Return: TechniqueResult with payload {"hits": List[Hit]}
"""
from collections.abc import Sequence
from typing import Any, List, Optional

from ..adapters.base import EmbedderAdapter, VectorStoreAdapter
from ..adapters.dummy_embedder import DummyEmbedder
from ..adapters.inmemory_vectorstore import InMemoryVectorStore
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Hit


@TechniqueRegistry.register
class DenseRetriever(RAGTechnique):
    meta = TechniqueMeta(
        name="Dense Retrieval / Vector Search (e.g., DPR)",
        category="core-retrieval",
        description="Production-friendly dense retriever with optional adapters fallback."
    )

    def __init__(
        self,
        embedder: Optional[EmbedderAdapter] = None,
        vectorstore: Optional[VectorStoreAdapter] = None,
        chunks: Optional[Sequence[Chunk]] = None,
        dim: int = 16,
    ):
        super().__init__(self.meta)
        # If adapters are not supplied, use internal defaults (so class works out-of-the-box)
        self.embedder = embedder if embedder is not None else DummyEmbedder(dim=dim)
        self.vectorstore = vectorstore if vectorstore is not None else InMemoryVectorStore()
        # If chunks provided at init, add them to the vectorstore
        if chunks:
            self.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        ids = []
        texts = []
        metas = []
        for c in chunks:
            ids.append(c.id)
            texts.append(c.text)
            metas.append({"chunk": c, "document_id": c.document_id})
        vectors = self.embedder.embed(texts)
        self.vectorstore.add(ids, vectors, metas)

    def apply(self, *args, **kwargs) -> Any:
        """
        apply(*args, **kwargs) behavior:
            - If called as apply(chunks=[...]) it will add those chunks and return a success.
            - Typical query usage: apply(query="...", top_k=5)
        """
        # If first positional arg is a list of chunks, treat it as an add request
        if args:
            first = args[0]
            if isinstance(first, (list, tuple)) and first and hasattr(first[0], "text"):
                self.add_chunks(first)
                return TechniqueResult(success=True, payload={"added": len(first)})
        # Recognize kwargs or second arg as query
        query = kwargs.pop("query", "") if "query" in kwargs else (args[1] if len(args) > 1 else "")
        top_k = kwargs.pop("top_k", 5)
        if not query:
            return TechniqueResult(success=True, payload={"hits": []})
        qvecs = self.embedder.embed([query])
        if not qvecs:
            return TechniqueResult(success=True, payload={"hits": []})
        qvec = qvecs[0]
        results = self.vectorstore.search(qvec, top_k=top_k)
        hits: List[Hit] = []
        for _id, score, meta in results:
            chunk_obj = None
            doc_id = None
            if meta and isinstance(meta, dict):
                chunk_obj = meta.get("chunk")
                doc_id = meta.get("document_id")
            hits.append(Hit(doc_id=doc_id or _id, score=float(score), chunk=chunk_obj, meta=meta or {}))
        return TechniqueResult(success=True, payload={"hits": hits})
