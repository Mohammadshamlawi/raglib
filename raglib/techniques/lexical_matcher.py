"""Lexical Matching retrieval technique.

A simple lexical matching implementation for text retrieval. Uses exact
and fuzzy string matching with configurable similarity thresholds.
Supports various matching modes including exact, substring, and token overlap.

Returns TechniqueResult with payload {"hits": List[Hit]} where Hit contains
document IDs and lexical matching scores.
"""
import re
from collections import Counter
from collections.abc import Sequence
from typing import Union

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Document, Hit

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization using regex."""
    return _WORD_RE.findall(text.lower())


def _exact_match_score(query: str, doc_text: str) -> float:
    """Exact string matching score."""
    query_clean = query.lower().strip()
    doc_clean = doc_text.lower().strip()
    return 1.0 if query_clean in doc_clean else 0.0


def _substring_match_score(query: str, doc_text: str) -> float:
    """Substring matching with position-based scoring."""
    query_clean = query.lower().strip()
    doc_clean = doc_text.lower().strip()
    
    if not query_clean or not doc_clean:
        return 0.0
    
    if query_clean in doc_clean:
        # Score based on position (earlier matches score higher)
        position = doc_clean.find(query_clean)
        position_factor = 1.0 - (position / len(doc_clean))
        length_factor = len(query_clean) / len(doc_clean)
        return min(1.0, position_factor * 0.7 + length_factor * 0.3)
    
    return 0.0


def _token_overlap_score(query: str, doc_text: str) -> float:
    """Token overlap scoring (Jaccard similarity)."""
    query_tokens = set(_tokenize(query))
    doc_tokens = set(_tokenize(doc_text))
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    intersection = len(query_tokens & doc_tokens)
    union = len(query_tokens | doc_tokens)
    
    return intersection / union if union > 0 else 0.0


def _weighted_token_overlap_score(query: str, doc_text: str) -> float:
    """Weighted token overlap using term frequencies."""
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(doc_text)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    query_counter = Counter(query_tokens)
    doc_counter = Counter(doc_tokens)
    
    # Calculate weighted overlap
    overlap_score = 0.0
    total_query_weight = sum(query_counter.values())
    
    for token, query_count in query_counter.items():
        if token in doc_counter:
            doc_count = doc_counter[token]
            # Weight by query term frequency and document term frequency
            weight = query_count / total_query_weight
            token_score = min(query_count, doc_count) / max(query_count, doc_count)
            overlap_score += weight * token_score
    
    return overlap_score


@TechniqueRegistry.register
class LexicalMatcher(RAGTechnique):
    """Lexical Matching retrieval technique.
    
    Implements various lexical matching strategies for text retrieval.
    Supports exact matching, substring matching, and token overlap scoring.
    Uses dependency-free implementation with configurable matching modes.
    
    Category: Sparse / lexical retrieval family
    Inputs: plain text (queries and documents), optional document corpus
    Outputs: ranked document lists with lexical matching scores
    Vector store: Not required (uses string matching algorithms)
    """
    
    meta = TechniqueMeta(
        name="lexical_matcher",
        category="sparse_retrieval",
        description="Lexical matching retrieval with configurable matching modes"
    )
    
    def __init__(self,
                 docs: Sequence[Union[Document, str]] = None,
                 mode: str = "token_overlap",
                 threshold: float = 0.0):
        """Initialize Lexical Matcher.
        
        Args:
            docs: Optional initial corpus to index immediately
            mode: Matching mode ("exact", "substring", "token_overlap",
                  "weighted_overlap")
            threshold: Minimum similarity threshold for results
        """
        super().__init__(self.meta)
        self.mode = mode
        self.threshold = float(threshold)
        
        # Internal storage
        self._docs: list[Document] = []
        
        # Scoring function mapping
        self._score_functions = {
            "exact": _exact_match_score,
            "substring": _substring_match_score,
            "token_overlap": _token_overlap_score,
            "weighted_overlap": _weighted_token_overlap_score,
        }
        
        if mode not in self._score_functions:
            raise ValueError(f"Invalid mode: {mode}. Choose from: "
                           f"{list(self._score_functions.keys())}")
        
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
            
            self._docs.append(document)
    
    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply lexical matching retrieval.
        
        Args:
            corpus: Optional corpus to index for this query 
                    (first positional arg or kwarg)
            query: Query string to search (second positional arg or kwarg)
            top_k: Number of results to return (default: 5)
            threshold: Override threshold for this query
            
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
        threshold = kwargs.pop("threshold", self.threshold)
        
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
                meta={
                    "query": query, 
                    "corpus_size": 0, 
                    "error": "No documents indexed"
                }
            )
        
        # Get scoring function
        score_fn = self._score_functions[self.mode]
        
        # Score all documents
        scored_docs: list[tuple[int, float]] = []
        for doc_idx, doc in enumerate(self._docs):
            score = score_fn(query, doc.text)
            if score >= threshold:
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
                meta={
                    "lexical_score": score,
                    "matching_mode": self.mode,
                    "threshold": threshold
                }
            )
            hits.append(hit)
        
        return TechniqueResult(
            success=True,
            payload={"hits": hits},
            meta={
                "query": query,
                "corpus_size": len(self._docs),
                "top_k": top_k,
                "total_scored": len(scored_docs),
                "mode": self.mode,
                "threshold": threshold,
                "above_threshold": len(scored_docs)
            }
        )