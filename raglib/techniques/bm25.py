"""BM25 (Best Matching 25) retrieval technique.

A robust BM25 implementation for text retrieval. Provides both lightweight
in-memory indexing and runtime corpus support. Uses simple tokenization
and standard BM25 scoring formula.

Returns TechniqueResult with payload {"hits": List[Hit]} where Hit contains
document IDs and relevance scores.
"""
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Union

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Document, Hit

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization using regex."""
    return _WORD_RE.findall(text.lower())


@TechniqueRegistry.register
class BM25(RAGTechnique):
    """BM25 (Best Matching 25) retrieval technique.
    
    Implements the BM25 ranking function for text retrieval. Supports both
    pre-indexing documents at initialization and providing corpus at query time.
    Uses dependency-free implementation with simple tokenization.
    
    Category: Sparse / lexical retrieval family
    Inputs: plain text (queries and documents), optional document corpus
    Outputs: ranked document lists with relevance scores
    Vector store: Not required (uses inverted indices)
    """
    
    meta = TechniqueMeta(
        name="BM25 (Best Matching 25)",
        category="sparse-retrieval",
        description="BM25 ranking function for text retrieval with in-memory indexing"
    )
    
    def __init__(self, 
                 docs: Sequence[Union[Document, str]] = None, 
                 k1: float = 1.5, 
                 b: float = 0.75):
        """Initialize BM25 retriever.
        
        Args:
            docs: Optional initial corpus to index immediately
            k1: BM25 k1 parameter (term frequency saturation point)
            b: BM25 b parameter (length normalization factor)
        """
        super().__init__(self.meta)
        self.k1 = float(k1)
        self.b = float(b)
        
        # Internal index structures
        self._docs: list[Document] = []
        self._tf: list[dict[str, int]] = []  # term frequencies per document
        self._df: dict[str, int] = defaultdict(int)  # document frequencies
        self._doc_lens: list[int] = []  # document lengths in tokens
        self._avgdl: float = 0.0  # average document length
        
        if docs:
            self.index(docs)
    
    def index(self, docs: Sequence[Union[Document, str]]) -> None:
        """Index a corpus of documents.
        
        Args:
            docs: Sequence of Document objects or raw strings
        """
        for doc in docs:
            # Normalize to Document object
            if hasattr(doc, "text"):
                document = doc
            else:
                document = Document(id=f"doc_{len(self._docs)}", text=str(doc))
            
            # Tokenize and build term frequency map
            tokens = _tokenize(document.text)
            tf_map = defaultdict(int)
            for token in tokens:
                tf_map[token] += 1
            
            # Update document frequency counts
            for term in set(tokens):
                self._df[term] += 1
            
            # Store document and statistics
            self._tf.append(dict(tf_map))
            self._doc_lens.append(len(tokens))
            self._docs.append(document)
        
        # Recompute average document length
        if self._doc_lens:
            self._avgdl = sum(self._doc_lens) / len(self._doc_lens)
        else:
            self._avgdl = 0.0
    
    def _score_document(self, query_terms: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            doc_idx: Index of document to score
            
        Returns:
            BM25 relevance score
        """
        if doc_idx >= len(self._docs):
            return 0.0
            
        score = 0.0
        N = len(self._docs)  # total number of documents
        dl = self._doc_lens[doc_idx]  # document length
        tf_doc = self._tf[doc_idx]  # term frequencies for this document
        
        for term in query_terms:
            df = self._df.get(term, 0)  # document frequency
            if df == 0:
                continue  # term not in corpus
            
            # IDF component with smoothing
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            
            # TF component with BM25 normalization
            tf = tf_doc.get(term, 0)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(1.0, self._avgdl))
            numerator = tf * (self.k1 + 1)
            
            if denominator > 0:
                score += idf * (numerator / denominator)
        
        return score
    
    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply BM25 retrieval.
        
        Args:
            corpus: Optional corpus to index for this query (first positional arg or kwarg)
            query: Query string to search (second positional arg or kwarg)
            top_k: Number of results to return (default: 5)
            
        Returns:
            TechniqueResult with hits payload containing ranked results
        """
        # Extract arguments with flexible signature support
        corpus = kwargs.pop("corpus", None)
        if corpus is None and args:
            corpus = args[0]
            
        query = kwargs.pop("query", "")
        if not query and len(args) > 1:
            query = args[1]
        elif not query and len(args) == 1 and isinstance(args[0], str):
            # Single string argument is treated as query
            query = args[0]
            corpus = None
            
        top_k = kwargs.pop("top_k", 5)
        
        # Index corpus if provided
        if corpus is not None:
            self.index(corpus)
        
        # Handle empty query
        if not query or not isinstance(query, str):
            return TechniqueResult(
                success=True,
                payload={"hits": []},
                meta={"query": query, "corpus_size": len(self._docs)}
            )
        
        # Handle empty corpus
        if not self._docs:
            return TechniqueResult(
                success=True,
                payload={"hits": []},
                meta={"query": query, "corpus_size": 0, "error": "No documents indexed"}
            )
        
        # Tokenize query
        query_terms = _tokenize(query)
        if not query_terms:
            return TechniqueResult(
                success=True,
                payload={"hits": []},
                meta={"query": query, "corpus_size": len(self._docs)}
            )
        
        # Score all documents
        scored_docs: list[tuple[int, float]] = []
        for doc_idx in range(len(self._docs)):
            score = self._score_document(query_terms, doc_idx)
            scored_docs.append((doc_idx, score))
        
        # Sort by score (descending) and take top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Build Hit objects
        hits: list[Hit] = []
        for doc_idx, score in scored_docs[:top_k]:
            hit = Hit(
                doc_id=self._docs[doc_idx].id,
                score=float(score),
                chunk=None,  # Document-level retrieval
                meta={"bm25_score": score}
            )
            hits.append(hit)
        
        return TechniqueResult(
            success=True,
            payload={"hits": hits},
            meta={
                "query": query,
                "query_terms": query_terms,
                "corpus_size": len(self._docs),
                "top_k": top_k,
                "total_scored": len(scored_docs)
            }
        )