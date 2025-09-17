"""Lexical Transformer Retrieval technique.

A transformer-aware lexical retrieval approach that combines traditional
lexical matching with transformer-based token importance weighting.
This implementation simulates transformer attention patterns for lexical scoring.

Returns TechniqueResult with payload {"hits": List[Hit]} where Hit contains
document IDs and transformer-weighted lexical scores.
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


def _compute_position_weights(tokens: list[str]) -> dict[int, float]:
    """Compute position-based weights simulating transformer positional encoding."""
    weights = {}
    seq_len = len(tokens)
    
    for i in range(seq_len):
        # Simulate positional encoding with decay
        pos_weight = 1.0 / (1.0 + 0.1 * i)  # Earlier positions get higher weights
        
        # Add sinusoidal-like variation (simplified transformer PE)
        sin_component = math.sin(i / 10000.0)
        cos_component = math.cos(i / 10000.0)
        positional_factor = 0.5 + 0.3 * sin_component + 0.2 * cos_component
        
        weights[i] = pos_weight * positional_factor
    
    return weights


def _compute_attention_weights(query_tokens: list[str],
                              doc_tokens: list[str]) -> dict[str, float]:
    """Compute attention-like weights between query and document tokens."""
    if not query_tokens or not doc_tokens:
        return {}
    
    attention_scores = defaultdict(float)
    
    # Compute pairwise similarities (simplified attention)
    for q_token in query_tokens:
        for d_token in doc_tokens:
            # Exact match gets high attention
            if q_token == d_token:
                attention_scores[d_token] += 1.0
            
            # Prefix/suffix matching gets moderate attention
            elif (len(q_token) > 2 and len(d_token) > 2 and
                  (q_token.startswith(d_token[:2]) or
                   q_token.endswith(d_token[-2:]))):
                attention_scores[d_token] += 0.5
            
            # Length similarity gets small attention
            elif abs(len(q_token) - len(d_token)) <= 1:
                attention_scores[d_token] += 0.1
    
    # Normalize attention scores
    if attention_scores:
        max_score = max(attention_scores.values())
        if max_score > 0:
            attention_scores = {
                token: score / max_score
                for token, score in attention_scores.items()
            }
    
    return dict(attention_scores)


def _transformer_lexical_score(query: str, doc_text: str) -> float:
    """Compute transformer-aware lexical similarity score."""
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(doc_text)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    # Compute position weights for document
    pos_weights = _compute_position_weights(doc_tokens)
    
    # Compute attention weights
    attention_weights = _compute_attention_weights(query_tokens, doc_tokens)
    
    # Compute final score combining position and attention
    total_score = 0.0
    total_weight = 0.0
    
    for i, token in enumerate(doc_tokens):
        if token in attention_weights:
            pos_weight = pos_weights.get(i, 1.0)
            attention_weight = attention_weights[token]
            
            # Combine positional and attention weights
            combined_weight = 0.6 * attention_weight + 0.4 * pos_weight
            total_score += combined_weight
            total_weight += 1.0
    
    # Normalize by document length and query coverage
    if total_weight > 0:
        base_score = total_score / total_weight
        
        # Boost score based on query term coverage
        query_terms_found = sum(1 for q_token in query_tokens 
                               if q_token in doc_tokens)
        coverage_boost = query_terms_found / len(query_tokens)
        
        return base_score * (0.7 + 0.3 * coverage_boost)
    
    return 0.0


@TechniqueRegistry.register
class LexicalTransformer(RAGTechnique):
    """Lexical Transformer Retrieval technique.
    
    Implements transformer-aware lexical retrieval that combines traditional
    lexical matching with simulated transformer attention patterns and
    positional encoding effects. Provides improved lexical matching by
    considering token positions and attention-like scoring.
    
    Category: Sparse / lexical retrieval family (transformer-enhanced)
    Inputs: plain text (queries and documents), transformer-like attention
    Outputs: ranked document lists with transformer-weighted lexical scores
    Vector store: Not required (uses enhanced lexical matching)
    """
    
    meta = TechniqueMeta(
        name="lexical_transformer",
        category="sparse_retrieval",
        description="Transformer-aware lexical retrieval with attention weighting"
    )
    
    def __init__(self,
                 docs: Sequence[Union[Document, str]] = None,
                 attention_weight: float = 0.6,
                 position_weight: float = 0.4):
        """Initialize Lexical Transformer retriever.
        
        Args:
            docs: Optional initial corpus to index immediately
            attention_weight: Weight for attention-based scoring (0.0 to 1.0)
            position_weight: Weight for position-based scoring (0.0 to 1.0)
        """
        super().__init__(self.meta)
        self.attention_weight = float(attention_weight)
        self.position_weight = float(position_weight)
        
        # Normalize weights
        total_weight = self.attention_weight + self.position_weight
        if total_weight > 0:
            self.attention_weight /= total_weight
            self.position_weight /= total_weight
        
        # Internal storage
        self._docs: list[Document] = []
        
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
        """Apply lexical transformer retrieval.
        
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
        
        # Score all documents using transformer-aware lexical matching
        scored_docs: list[tuple[int, float]] = []
        for doc_idx, doc in enumerate(self._docs):
            score = _transformer_lexical_score(query, doc.text)
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
                        "transformer_lexical_score": score,
                        "attention_weight": self.attention_weight,
                        "position_weight": self.position_weight
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
                "attention_weight": self.attention_weight,
                "position_weight": self.position_weight
            }
        )