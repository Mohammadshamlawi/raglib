"""In-memory dense retriever using adapters.

This technique consumes an EmbedderAdapter and a VectorStoreAdapter to:
- embed chunks and add them to the vector store with metadata (store the Chunk in metadata)
- at query time embed the query and search the vectorstore, returning Hit objects.

It remains a one-file-per-technique implementation (registered via TechniqueRegistry).
"""
from typing import List, Optional, Sequence

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..schemas import Chunk, Hit
from ..registry import TechniqueRegistry
from ..adapters.base import EmbedderAdapter, VectorStoreAdapter


@TechniqueRegistry.register
class InMemoryDenseRetriever(RAGTechnique):
    meta = TechniqueMeta(
        name="inmemory_dense",
        category="core-retrieval",
        description="In-memory dense retriever using adapters for embedder and vectorstore."
    )

    def __init__(
        self,
        embedder: Optional[EmbedderAdapter] = None,
        vectorstore: Optional[VectorStoreAdapter] = None,
        chunks: Optional[Sequence[Chunk]] = None,
    ):
        super().__init__(self.meta)
        if embedder is None or vectorstore is None:
            raise ValueError("embedder and vectorstore adapters are required")
        self.embedder = embedder
        self.vectorstore = vectorstore
        if chunks:
            # add initial chunks
            self.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Embed and add chunks to the vectorstore. Metadata stores the original Chunk."""
        ids = []
        texts = []
        metas = []
        for c in chunks:
            ids.append(c.id)
            texts.append(c.text)
            # store the chunk object in metadata so we can reconstruct Hits
            metas.append({"chunk": c, "document_id": c.document_id})
        vectors = self.embedder.embed(texts)
        self.vectorstore.add(ids, vectors, metas)

    def apply(self, query: str, top_k: int = 5) -> TechniqueResult:
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
