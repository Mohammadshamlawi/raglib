"""CrossEncoder re-ranking technique.

This module implements cross-encoder re-ranking that scores (query, document) pairs.
Supports both LLM adapter-backed and deterministic lexical fallback modes.
"""

from typing import Any, Optional, List, Callable
import re
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Hit


def _tokenize(text: str) -> List[str]:
    """Simple tokenization: extract words, convert to lowercase."""
    return re.findall(r'\w+', text.lower())


def _lexical_overlap_score(query: str, doc_text: str) -> float:
    """Deterministic lexical overlap scorer.
    
    Returns normalized count of shared tokens between query and document.
    """
    if not query or not doc_text:
        return 0.0
        
    query_tokens = set(_tokenize(query))
    doc_tokens = set(_tokenize(doc_text))
    
    if not query_tokens:
        return 0.0
        
    overlap = len(query_tokens & doc_tokens)
    return overlap / len(query_tokens)


def _parse_score_from_llm_output(output: str) -> float:
    """Parse numeric score from LLM output.
    
    Looks for float values in the output, returns the first one found.
    If no float is found, attempts to map ordinal terms to scores.
    Falls back to 0.0 if parsing fails.
    """
    # Try to extract a float
    import re
    float_matches = re.findall(r'\d*\.?\d+', output)
    if float_matches:
        try:
            score = float(float_matches[0])
            # Clamp to reasonable range
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
    
    # Map ordinal terms to scores (check longer phrases first)
    output_lower = output.lower()
    if any(phrase in output_lower for phrase in ['excellent', 'perfect', 'very high']):
        return 0.9
    elif any(phrase in output_lower for phrase in ['very low', 'terrible', 'irrelevant']):
        return 0.1
    elif any(phrase in output_lower for phrase in ['good', 'high', 'relevant']):
        return 0.7
    elif any(phrase in output_lower for phrase in ['medium', 'moderate', 'okay']):
        return 0.5
    elif any(phrase in output_lower for phrase in ['low', 'poor', 'bad']):
        return 0.3
    
    # Default fallback
    return 0.0


@TechniqueRegistry.register
class CrossEncoderReRanker(RAGTechnique):
    """Cross-encoder re-ranker that scores (query, document) pairs.
    
    Args:
        llm_adapter: Optional LLM adapter for scoring pairs
        scoring_prompt_template: Template for LLM scoring prompts
        fallback_scoring_fn: Optional deterministic fallback scorer
    """
    
    meta = TechniqueMeta(
        name="crossencoder_reranker",
        category="reranking", 
        description="Cross-encoder re-ranking using pairwise (query, document) scoring"
    )

    def __init__(self,
                 llm_adapter: Optional[Any] = None,
                 scoring_prompt_template: str = "Score the relevance of the document to the query: {query} ||| {doc}",
                 fallback_scoring_fn: Optional[Callable[[str, str], float]] = None):
        super().__init__(self.meta)
        self.llm_adapter = llm_adapter
        self.scoring_prompt_template = scoring_prompt_template
        self.fallback_scoring_fn = fallback_scoring_fn or _lexical_overlap_score

    def _get_document_text(self, hit: Hit) -> str:
        """Extract text content from a hit."""
        if hit.chunk and hit.chunk.text:
            return hit.chunk.text
        elif "text" in hit.meta:
            return hit.meta["text"]
        elif "content" in hit.meta:
            return hit.meta["content"]
        return ""

    def _score_pair(self, query: str, doc_text: str) -> float:
        """Score a (query, document) pair."""
        if self.llm_adapter is not None:
            # Use LLM adapter for scoring
            prompt = self.scoring_prompt_template.format(query=query, doc=doc_text)
            try:
                llm_output = self.llm_adapter.generate(prompt)
                return _parse_score_from_llm_output(llm_output)
            except Exception:
                # Fall back to lexical scoring on error
                return self.fallback_scoring_fn(query, doc_text)
        else:
            # Use fallback lexical scoring
            return self.fallback_scoring_fn(query, doc_text)

    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply cross-encoder re-ranking.
        
        Expected inputs:
            hits: List[Hit] - candidate documents to re-rank
            query: str - the user query
            top_k: Optional[int] - number of results to return (default: 5)
        """
        # Extract parameters
        hits = None
        query = None
        top_k = 5
        
        # Handle positional args
        if len(args) >= 1:
            hits = args[0]
        if len(args) >= 2:
            query = args[1]
        if len(args) >= 3:
            top_k = args[2]
            
        # Handle keyword args
        hits = kwargs.get('hits', hits)
        query = kwargs.get('query', query)
        top_k = kwargs.get('top_k', top_k)
        
        # Validate inputs
        if not hits:
            return TechniqueResult(success=True, payload={"hits": []})
            
        if not query:
            # If no query, return hits unchanged (or empty)
            return TechniqueResult(success=True, payload={"hits": hits[:top_k]})
        
        # Score each hit
        scored_hits = []
        for hit in hits:
            doc_text = self._get_document_text(hit)
            score = self._score_pair(query, doc_text)
            
            # Create a new hit with cross-encoder score
            # Store original score and add cross-encoder score to meta
            new_hit = Hit(
                doc_id=hit.doc_id,
                score=score,  # Replace with cross-encoder score
                chunk=hit.chunk,
                meta={
                    **hit.meta,
                    "original_score": hit.score,
                    "cross_score": score
                }
            )
            scored_hits.append(new_hit)
        
        # Sort by cross-encoder score (descending) with deterministic tie-breaking
        scored_hits.sort(key=lambda h: (-h.score, h.doc_id))
        
        # Return top_k results
        selected_hits = scored_hits[:top_k]
        
        return TechniqueResult(
            success=True,
            payload={"hits": selected_hits},
            meta={
                "algorithm": "cross_encoder",
                "scoring_method": "llm_adapter" if self.llm_adapter else "lexical_fallback",
                "selected_count": len(selected_hits)
            }
        )
