"""Simple BM25 retriever (toy implementation, dependency-free).

Implements a minimal BM25-style retriever for testing and development. Document
inputs may be plain strings or `Document` dataclasses. Uses simple tokenization
via regex to split words and a common BM25 scoring formula.

Returns a TechniqueResult with payload {"hits": List[Hit]} where Hit.doc_id is set
and Hit.chunk is None for doc-level retrieval.
"""
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple, Union

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Document, Hit

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


@TechniqueRegistry.register
class BM25Simple(RAGTechnique):
    meta = TechniqueMeta(
        name="bm25_simple",
        category="core-retrieval",
        description="Toy BM25 retriever (pure python, dependency-free)."
    )

    def __init__(self, docs: Sequence[Union[Document, str]] = None, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            docs: optional initial corpus (list of Document or raw string). If provided,
                  the retriever indexes them immediately.
            k1, b: BM25 hyperparameters.
        """
        super().__init__(self.meta)
        self.k1 = float(k1)
        self.b = float(b)
        self._docs: List[Document] = []
        # internal indices
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0
        if docs:
            self.index(docs)

    def index(self, docs: Sequence[Union[Document, str]]) -> None:
        """Index a corpus of documents."""
        for i, doc in enumerate(docs):
            if hasattr(doc, "text"):
                d = doc
            else:
                d = Document(id=f"doc_{len(self._docs)}", text=str(doc))
            tokens = _tokenize(d.text)
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            # update df
            for t in set(tokens):
                self._df[t] += 1
            self._tf.append(dict(tf))
            self._doc_lens.append(len(tokens))
            self._docs.append(d)
        # recompute avgdl
        if self._doc_lens:
            self._avgdl = sum(self._doc_lens) / len(self._doc_lens)
        else:
            self._avgdl = 0.0

    def _score_doc(self, query_terms: List[str], doc_idx: int) -> float:
        score = 0.0
        N = len(self._docs)
        dl = self._doc_lens[doc_idx]
        tf_doc = self._tf[doc_idx]
        for term in query_terms:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            # IDF with smoothing
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            tf = tf_doc.get(term, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1.0, self._avgdl))
            numer = tf * (self.k1 + 1)
            score += idf * (numer / denom) if denom > 0 else 0.0
        return score

    def apply(self, corpus: Any = None, query: str = "", top_k: int = 5) -> TechniqueResult:
        """
        Args:
            corpus: optional list of Document or strings to index for this call. If None,
                    uses previously indexed documents.
            query: query string to search.
            top_k: number of results to return.

        Returns:
            TechniqueResult with payload {"hits": List[Hit]}
        """
        if corpus is not None:
            # allow passing corpus inline
            self.index(corpus)

        if not query:
            return TechniqueResult(success=True, payload={"hits": []})

        tokens = _tokenize(query)
        scores: List[Tuple[int, float]] = []
        for idx in range(len(self._docs)):
            sc = self._score_doc(tokens, idx)
            scores.append((idx, sc))
        # sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        hits: List[Hit] = []
        for idx, sc in scores[:top_k]:
            hits.append(Hit(doc_id=self._docs[idx].id, score=float(sc), chunk=None, meta={}))
        return TechniqueResult(success=True, payload={"hits": hits})
