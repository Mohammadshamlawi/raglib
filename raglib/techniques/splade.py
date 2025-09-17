"""SPLADE (Sparse Lexical and Expansion model for Dense information retrieval).

A hybrid sparse-dense retrieval technique that combines learned sparse
representations with dense embeddings. This is a simplified implementation
that focuses on sparse term expansion and weighting.

Returns TechniqueResult with payload {"hits": List[Hit]} where Hit contains
document IDs and SPLADE-style scores.
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


def _compute_term_importance(text: str, vocabulary: dict[str, int]) -> dict[str, float]:
    """Compute term importance using TF-IDF with expansion weights.
    
    This is a simplified version of SPLADE's learned term importance.
    In practice, SPLADE uses transformer models to predict term weights.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}
    
    # Compute term frequencies
    tf_map = defaultdict(int)
    for token in tokens:
        tf_map[token] += 1
    
    # Compute importance scores (simplified SPLADE-style)
    importance_scores = {}
    total_tokens = len(tokens)
    
    for term, tf in tf_map.items():
        # Base TF score
        tf_score = tf / total_tokens
        
        # Add expansion factor based on term rarity
        vocab_size = len(vocabulary)
        df = vocabulary.get(term, 1)
        idf_factor = math.log(vocab_size / df) if df > 0 else 0
        
        # SPLADE-style importance (simplified)
        # In real SPLADE, this would be learned by a transformer
        importance = tf_score * (1 + 0.1 * idf_factor)
        
        # Apply sparsity regularization (keep only significant terms)
        if importance > 0.01:  # Threshold for sparsity
            importance_scores[term] = importance
    
    return importance_scores


def _expand_query_terms(query: str, vocabulary: dict[str, int]) -> dict[str, float]:
    """Expand query terms with related terms (simplified expansion).
    
    In real SPLADE, this uses learned embeddings to find related terms.
    Here we use simple heuristics for demonstration.
    """
    query_tokens = _tokenize(query)
    expanded_terms = {}
    
    # Add original query terms with high weight
    for token in query_tokens:
        expanded_terms[token] = expanded_terms.get(token, 0.0) + 1.0
    
    # Simple expansion: add terms that share prefixes or suffixes
    for token in query_tokens:
        if len(token) > 3:
            prefix = token[:3]
            suffix = token[-3:]
            
            for vocab_term in vocabulary:
                if vocab_term != token and len(vocab_term) > 3:
                    # Prefix/suffix matching for expansion
                    if vocab_term.startswith(prefix) or vocab_term.endswith(suffix):
                        expansion_weight = 0.3  # Weight for expanded terms
                        expanded_terms[vocab_term] = max(
                            expanded_terms.get(vocab_term, 0.0),
                            expansion_weight
                        )
    
    return expanded_terms


def _splade_similarity(query_terms: dict[str, float],
                      doc_terms: dict[str, float]) -> float:
    """Compute SPLADE-style similarity between sparse representations."""
    if not query_terms or not doc_terms:
        return 0.0
    
    # Compute dot product of sparse vectors
    score = 0.0
    for term, query_weight in query_terms.items():
        if term in doc_terms:
            score += query_weight * doc_terms[term]
    
    return score


@TechniqueRegistry.register
class Splade(RAGTechnique):
    """SPLADE (SParse Lexical And Dense Expansion) retrieval technique.
    
    Implements a simplified version of SPLADE that combines sparse lexical
    matching with term expansion. Uses learned term importance and query
    expansion to improve retrieval performance over traditional sparse methods.
    
    Category: Sparse / lexical retrieval family (hybrid)
    Inputs: plain text (queries and documents), learned term weights
    Outputs: ranked document lists with SPLADE-style scores
    Vector store: Not required (uses sparse representations)
    """
    
    meta = TechniqueMeta(
        name="splade",
        category="sparse_retrieval",
        description="SPLADE sparse-dense hybrid retrieval with term expansion"
    )
    
    def __init__(self,
                 docs: Sequence[Union[Document, str]] = None,
                 expansion_factor: float = 0.3,
                 sparsity_threshold: float = 0.01):
        """Initialize SPLADE retriever.
        
        Args:
            docs: Optional initial corpus to index immediately
            expansion_factor: Weight for expanded terms (0.0 to 1.0)
            sparsity_threshold: Minimum importance for term inclusion
        """
        super().__init__(self.meta)
        self.expansion_factor = float(expansion_factor)
        self.sparsity_threshold = float(sparsity_threshold)
        
        # Internal index structures
        self._docs: list[Document] = []
        self._vocabulary: dict[str, int] = defaultdict(int)  # term -> doc frequency
        self._doc_representations: list[dict[str, float]] = []  # sparse vectors
        
        if docs:
            self.index(docs)
    
    def index(self, docs: Sequence[Union[Document, str]]) -> None:
        """Index a corpus of documents with SPLADE representations.
        
        Args:
            docs: Sequence of Document objects or raw strings
        """
        # First pass: build vocabulary
        for doc in docs:
            if hasattr(doc, "text"):
                text = doc.text
            else:
                text = str(doc)
            
            tokens = set(_tokenize(text))
            for token in tokens:
                self._vocabulary[token] += 1
        
        # Second pass: compute document representations
        for doc in docs:
            # Normalize to Document object
            if hasattr(doc, "text"):
                document = doc
            else:
                document = Document(id=f"doc_{len(self._docs)}", text=str(doc))
            
            # Compute SPLADE representation
            doc_representation = _compute_term_importance(
                document.text, 
                dict(self._vocabulary)
            )
            
            # Apply sparsity threshold
            doc_representation = {
                term: weight 
                for term, weight in doc_representation.items()
                if weight >= self.sparsity_threshold
            }
            
            self._docs.append(document)
            self._doc_representations.append(doc_representation)
    
    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply SPLADE retrieval.
        
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
                meta={
                    "query": query,
                    "corpus_size": 0,
                    "error": "No documents indexed"
                }
            )
        
        # Expand query terms
        expanded_query = _expand_query_terms(query, dict(self._vocabulary))
        if not expanded_query:
            return TechniqueResult(
                success=True,
                payload={"hits": []},
                meta={
                    "query": query,
                    "corpus_size": len(self._docs),
                    "error": "No valid query terms"
                }
            )
        
        # Score all documents using SPLADE similarity
        scored_docs: list[tuple[int, float]] = []
        for doc_idx, doc_repr in enumerate(self._doc_representations):
            score = _splade_similarity(expanded_query, doc_repr)
            scored_docs.append((doc_idx, score))
        
        # Sort by score (descending) and take top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Build Hit objects
        hits: list[Hit] = []
        for doc_idx, score in scored_docs[:top_k]:
            if score > 0.0:  # Only include documents with positive scores
                hit = Hit(
                    doc_id=self._docs[doc_idx].id,
                    score=float(score),
                    chunk=None,  # Document-level retrieval
                    meta={
                        "splade_score": score,
                        "expansion_factor": self.expansion_factor,
                        "sparsity_threshold": self.sparsity_threshold
                    }
                )
                hits.append(hit)
        
        return TechniqueResult(
            success=True,
            payload={"hits": hits},
            meta={
                "query": query,
                "expanded_terms": list(expanded_query.keys()),
                "corpus_size": len(self._docs),
                "top_k": top_k,
                "total_scored": len(scored_docs),
                "vocabulary_size": len(self._vocabulary),
                "expansion_factor": self.expansion_factor
            }
        )