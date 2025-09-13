"""Production-friendly BM25 retriever.

This class provides a production-style name and behavior while preserving
the `apply(self, *args, **kwargs)` contract.

Usage:
    retriever = BM25Retriever()  # no corpus indexed yet
    result = retriever.apply(corpus=my_docs, query="banana", top_k=5)

If you prefer pre-indexing:
    retriever = BM25Retriever(docs=my_docs)
    result = retriever.apply(query="banana", top_k=5)
"""
from typing import Any, Sequence, Union
from ..core import RAGTechnique, TechniqueMeta
from ..schemas import Document
from ..registry import TechniqueRegistry
from .bm25_simple import BM25Simple  # reuse toy BM25 implementation internally


@TechniqueRegistry.register
class BM25Retriever(RAGTechnique):
    """Production-friendly BM25 retriever wrapper.

    Internally uses the BM25Simple implementation for text-only retrieval.
    Works without any external adapters. The apply signature supports being
    passed a corpus at call time or using pre-indexed docs at construction.
    """

    meta = TechniqueMeta(
        name="bm25_retriever",
        category="core-retrieval",
        description="Production-friendly BM25 retriever (wrapper over BM25Simple)."
    )

    def __init__(self, docs: Sequence[Union[Document, str]] = None, k1: float = 1.5, b: float = 0.75):
        super().__init__(self.meta)
        # Keep an internal BM25Simple instance which can be created empty
        self._bm25 = BM25Simple(docs=docs, k1=k1, b=b)

    def apply(self, *args, **kwargs) -> Any:
        """
        Apply contract: apply(*args, **kwargs) -> TechniqueResult

        Recognized kwargs:
            - corpus: optional Sequence[Document|str] to index for this call
            - query: the query string
            - top_k: results count
        """
        corpus = kwargs.pop("corpus", None) if "corpus" in kwargs else (args[0] if args else None)
        query = kwargs.pop("query", "") if "query" in kwargs else (args[1] if len(args) > 1 else "")
        top_k = kwargs.pop("top_k", 5)

        # If corpus was supplied, index it temporarily (BM25Simple.index appends)
        if corpus:
            self._bm25.index(corpus)

        # Delegate to BM25Simple apply (it returns TechniqueResult)
        return self._bm25.apply(corpus=None, query=query, top_k=top_k)
