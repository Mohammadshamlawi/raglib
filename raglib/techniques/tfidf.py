"""TF-IDF (Term Frequency - Inverse Document Frequency) retrieval technique.

A classic TF-IDF implementation for text retrieval. Provides sparse vector
representation and cosine similarity scoring. Uses simple tokenization
and standard TF-IDF weighting schemes.

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


def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    if not vec1 or not vec2:
        return 0.0
    
    # Compute dot product
    dot_product = 0.0
    for term in vec1:
        if term in vec2:
            dot_product += vec1[term] * vec2[term]
    
    if dot_product == 0.0:
        return 0.0
    
    # Compute magnitudes
    mag1 = math.sqrt(sum(val * val for val in vec1.values()))
    mag2 = math.sqrt(sum(val * val for val in vec2.values()))
    
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


@TechniqueRegistry.register
class TfIdf(RAGTechnique):
    """TF-IDF (Term Frequency - Inverse Document Frequency) retrieval technique.
    
    Implements classic TF-IDF scoring for text retrieval. Supports both
    pre-indexing documents at initialization and providing corpus at query time.
    Uses cosine similarity for scoring and dependency-free implementation.
    
    Category: Sparse / lexical retrieval family
    Inputs: plain text (queries and documents), optional document corpus
    Outputs: ranked document lists with cosine similarity scores
    Vector store: Not required (uses sparse vector representation)
    """
    
    meta = TechniqueMeta(
        name="tfidf",
        category="sparse_retrieval",
        description="TF-IDF retrieval with cosine similarity scoring"
    )
    
    def __init__(self,
                 docs: Sequence[Union[Document, str]] = None,
                 norm: str = "l2"):
        """Initialize TF-IDF retriever.
        
        Args:
            docs: Optional initial corpus to index immediately
            norm: Normalization scheme ("l2" for L2 norm, "none" for no normalization)
        """
        super().__init__(self.meta)
        self.norm = norm
        
        # Internal index structures
        self._docs: list[Document] = []
        self._tf: list[dict[str, int]] = []  # term frequencies per document
        self._df: dict[str, int] = defaultdict(int)  # document frequencies
        self._doc_vectors: list[dict[str, float]] = []  # TF-IDF vectors per document
        self._vocabulary: set[str] = set()
        
        if docs:
            self.index(docs)
    
    def index(self, docs: Sequence[Union[Document, str]]) -> None:
        """Index a corpus of documents.
        
        Args:
            docs: Sequence of Document objects or raw strings
        """
        # Reset if re-indexing
        
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
                self._vocabulary.add(token)
            
            # Update document frequency counts
            for term in set(tokens):
                self._df[term] += 1
            
            # Store document and statistics
            self._tf.append(dict(tf_map))
            self._docs.append(document)
        
        # Recompute TF-IDF vectors for all documents
        self._compute_tfidf_vectors()
    
    def _compute_tfidf_vectors(self) -> None:
        """Compute TF-IDF vectors for all indexed documents."""
        N = len(self._docs)
        self._doc_vectors = []
        
        for doc_idx in range(N):
            tf_doc = self._tf[doc_idx]
            tfidf_vector = {}
            
            for term, tf in tf_doc.items():
                df = self._df[term]
                if df > 0:
                    # TF-IDF = TF * IDF with smoothing
                    idf = math.log(N / df) + 1  # Add 1 for smoothing
                    tfidf_score = tf * idf
                    tfidf_vector[term] = tfidf_score
            
            # Apply normalization
            if self.norm == "l2" and tfidf_vector:
                magnitude = math.sqrt(sum(val * val for val in tfidf_vector.values()))
                if magnitude > 0:
                    tfidf_vector = {
                        term: val / magnitude
                        for term, val in tfidf_vector.items()
                    }
            
            self._doc_vectors.append(tfidf_vector)
    
    def _compute_query_vector(self, query_terms: list[str]) -> dict[str, float]:
        """Compute TF-IDF vector for query terms.
        
        Args:
            query_terms: List of query terms
            
        Returns:
            Query TF-IDF vector
        """
        if not self._docs:
            return {}
        
        N = len(self._docs)
        query_tf = defaultdict(int)
        
        # Compute query term frequencies
        for term in query_terms:
            query_tf[term] += 1
        
        # Compute query TF-IDF vector
        query_vector = {}
        for term, tf in query_tf.items():
            df = self._df.get(term, 0)
            if df > 0:
                idf = math.log(N / df) + 1  # Add 1 for smoothing
                tfidf_score = tf * idf
                query_vector[term] = tfidf_score
        
        # Apply normalization
        if self.norm == "l2" and query_vector:
            magnitude = math.sqrt(sum(val * val for val in query_vector.values()))
            if magnitude > 0:
                query_vector = {
                    term: val / magnitude
                    for term, val in query_vector.items()
                }
        
        return query_vector
    
    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply TF-IDF retrieval.
        
        Args:
            corpus: Optional corpus to index for this query
                    (first positional arg or kwarg)
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
        
        # Compute query vector
        query_vector = self._compute_query_vector(query_terms)
        if not query_vector:
            return TechniqueResult(
                success=True,
                payload={"hits": []},
                meta={
                    "query": query,
                    "corpus_size": len(self._docs),
                    "error": "No query terms in vocabulary"
                }
            )
        
        # Score all documents using cosine similarity
        scored_docs: list[tuple[int, float]] = []
        for doc_idx, doc_vector in enumerate(self._doc_vectors):
            score = _cosine_similarity(query_vector, doc_vector)
            scored_docs.append((doc_idx, score))
        
        # Sort by score (descending) and take top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Build Hit objects
        hits: list[Hit] = []
        for doc_idx, score in scored_docs[:top_k]:
            if score > 0.0:  # Only include documents with positive similarity
                hit = Hit(
                    doc_id=self._docs[doc_idx].id,
                    score=float(score),
                    chunk=None,  # Document-level retrieval
                    meta={"tfidf_score": score, "cosine_similarity": score}
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
                "total_scored": len(scored_docs),
                "vocabulary_size": len(self._vocabulary)
            }
        )
