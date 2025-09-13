"""Tests for CrossEncoder re-ranker."""

from raglib.techniques.crossencoder_rerank import CrossEncoderReRanker
from raglib.schemas import Hit, Chunk
from raglib.core import TechniqueResult


class DummyLLM:
    """Mock LLM adapter for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Return deterministic responses based on prompt content."""
        self.call_count += 1
        
        # Return pre-configured responses if available
        if prompt in self.responses:
            return self.responses[prompt]
            
        # Fallback deterministic responses based on content
        if "python" in prompt.lower() and "programming" in prompt.lower():
            return "0.9"  # High relevance
        elif "java" in prompt.lower() and "programming" in prompt.lower():
            return "0.8"  # Good relevance  
        elif "machine learning" in prompt.lower():
            return "0.7"  # Medium relevance
        elif "cooking" in prompt.lower():
            return "0.3"  # Low relevance
        else:
            return "0.5"  # Default medium


def test_crossencoder_fallback_lexical():
    """Test CrossEncoder with fallback lexical scorer."""
    hits = [
        Hit(doc_id="d1", score=0.9, chunk=Chunk(id="c1", document_id="d1", text="python programming language", start_idx=0, end_idx=26)),
        Hit(doc_id="d2", score=0.8, chunk=Chunk(id="c2", document_id="d2", text="java programming language", start_idx=0, end_idx=24)),
        Hit(doc_id="d3", score=0.7, chunk=Chunk(id="c3", document_id="d3", text="cooking recipes and food", start_idx=0, end_idx=24)),
        Hit(doc_id="d4", score=0.6, chunk=Chunk(id="c4", document_id="d4", text="python snake animal", start_idx=0, end_idx=19)),
    ]
    
    reranker = CrossEncoderReRanker()  # No LLM adapter
    result = reranker.apply(hits=hits, query="python programming", top_k=3)
    
    assert isinstance(result, TechniqueResult)
    assert result.success is True
    assert "hits" in result.payload
    
    ranked_hits = result.payload["hits"]
    assert len(ranked_hits) <= 3
    
    # d1 should rank highest (has both "python" and "programming")
    assert ranked_hits[0].doc_id == "d1"
    
    # Check that cross_score is in metadata
    for hit in ranked_hits:
        assert "cross_score" in hit.meta
        assert "original_score" in hit.meta


def test_crossencoder_deterministic():
    """Test that CrossEncoder produces deterministic results."""
    hits = [
        Hit(doc_id="d1", score=0.9, chunk=Chunk(id="c1", document_id="d1", text="apple fruit red", start_idx=0, end_idx=15)),
        Hit(doc_id="d2", score=0.8, chunk=Chunk(id="c2", document_id="d2", text="banana fruit yellow", start_idx=0, end_idx=19)),
        Hit(doc_id="d3", score=0.7, chunk=Chunk(id="c3", document_id="d3", text="car vehicle transport", start_idx=0, end_idx=21)),
    ]
    
    reranker = CrossEncoderReRanker()
    
    result1 = reranker.apply(hits=hits, query="fruit", top_k=2)
    result2 = reranker.apply(hits=hits, query="fruit", top_k=2)
    
    hits1 = result1.payload["hits"]
    hits2 = result2.payload["hits"]
    
    # Results should be identical
    assert len(hits1) == len(hits2)
    for h1, h2 in zip(hits1, hits2):
        assert h1.doc_id == h2.doc_id
        assert h1.score == h2.score


def test_crossencoder_with_llm_adapter():
    """Test CrossEncoder with LLM adapter."""
    hits = [
        Hit(doc_id="d1", score=0.5, chunk=Chunk(id="c1", document_id="d1", text="python programming tutorial", start_idx=0, end_idx=26)),
        Hit(doc_id="d2", score=0.5, chunk=Chunk(id="c2", document_id="d2", text="java programming guide", start_idx=0, end_idx=22)),
        Hit(doc_id="d3", score=0.5, chunk=Chunk(id="c3", document_id="d3", text="cooking recipes", start_idx=0, end_idx=15)),
    ]
    
    dummy_llm = DummyLLM()
    reranker = CrossEncoderReRanker(llm_adapter=dummy_llm)
    
    result = reranker.apply(hits=hits, query="programming", top_k=3)
    
    assert isinstance(result, TechniqueResult)
    assert result.success is True
    ranked_hits = result.payload["hits"]
    
    # Python should rank higher than Java (0.9 vs 0.8) based on DummyLLM logic
    doc_ids = [hit.doc_id for hit in ranked_hits]
    python_idx = doc_ids.index("d1")
    java_idx = doc_ids.index("d2")
    assert python_idx < java_idx, "Python should rank higher than Java"
    
    # Cooking should rank lowest
    cooking_idx = doc_ids.index("d3")
    assert cooking_idx > python_idx and cooking_idx > java_idx, "Cooking should rank lowest"


def test_crossencoder_empty_hits():
    """Test CrossEncoder with empty hits list."""
    reranker = CrossEncoderReRanker()
    result = reranker.apply(hits=[], query="test", top_k=5)
    
    assert isinstance(result, TechniqueResult)
    assert result.success is True
    assert result.payload["hits"] == []


def test_crossencoder_empty_query():
    """Test CrossEncoder with empty query."""
    hits = [
        Hit(doc_id="d1", score=0.9, chunk=Chunk(id="c1", document_id="d1", text="some text", start_idx=0, end_idx=9)),
        Hit(doc_id="d2", score=0.8, chunk=Chunk(id="c2", document_id="d2", text="other text", start_idx=0, end_idx=10)),
    ]
    
    reranker = CrossEncoderReRanker()
    result = reranker.apply(hits=hits, query="", top_k=5)
    
    assert isinstance(result, TechniqueResult)
    assert result.success is True
    # Should return hits unchanged when query is empty
    assert len(result.payload["hits"]) <= 5


def test_crossencoder_custom_scoring_template():
    """Test CrossEncoder with custom scoring template."""
    hits = [
        Hit(doc_id="d1", score=0.5, chunk=Chunk(id="c1", document_id="d1", text="machine learning algorithms", start_idx=0, end_idx=26)),
    ]
    
    # Custom responses for custom template
    dummy_llm = DummyLLM(responses={
        "Rate relevance: machine learning ||| machine learning algorithms": "excellent"
    })
    
    reranker = CrossEncoderReRanker(
        llm_adapter=dummy_llm,
        scoring_prompt_template="Rate relevance: {query} ||| {doc}"
    )
    
    result = reranker.apply(hits=hits, query="machine learning", top_k=1)
    
    assert isinstance(result, TechniqueResult)
    assert result.success is True
    ranked_hits = result.payload["hits"]
    
    # Should get high score for "excellent" response
    assert ranked_hits[0].score == 0.9


def test_crossencoder_hit_text_extraction():
    """Test different ways of extracting text from hits."""
    hits = [
        # Text in chunk
        Hit(doc_id="d1", score=0.5, chunk=Chunk(id="c1", document_id="d1", text="chunk text", start_idx=0, end_idx=10)),
        # Text in meta
        Hit(doc_id="d2", score=0.5, meta={"text": "meta text"}),
        # Content in meta  
        Hit(doc_id="d3", score=0.5, meta={"content": "content text"}),
        # No text
        Hit(doc_id="d4", score=0.5),
    ]
    
    reranker = CrossEncoderReRanker()
    result = reranker.apply(hits=hits, query="text", top_k=4)
    
    assert isinstance(result, TechniqueResult)
    assert result.success is True
    ranked_hits = result.payload["hits"]
    
    # All hits should have been processed (even the one with no text)
    assert len(ranked_hits) == 4


def test_crossencoder_score_parsing():
    """Test score parsing from LLM output."""
    from raglib.techniques.crossencoder_rerank import _parse_score_from_llm_output
    
    # Test float extraction
    assert _parse_score_from_llm_output("The score is 0.75") == 0.75
    assert _parse_score_from_llm_output("0.9") == 0.9
    assert _parse_score_from_llm_output("Score: 1.2 out of 1.0") == 1.0  # Clamped
    
    # Test ordinal mapping
    assert _parse_score_from_llm_output("excellent match") == 0.9
    assert _parse_score_from_llm_output("good relevance") == 0.7
    assert _parse_score_from_llm_output("medium quality") == 0.5
    assert _parse_score_from_llm_output("low relevance") == 0.3
    assert _parse_score_from_llm_output("very low quality") == 0.1
    
    # Test fallback
    assert _parse_score_from_llm_output("no clear indication") == 0.0


def test_crossencoder_with_positional_args():
    """Test CrossEncoder with positional arguments."""
    hits = [
        Hit(doc_id="d1", score=0.9, chunk=Chunk(id="c1", document_id="d1", text="python code", start_idx=0, end_idx=11)),
        Hit(doc_id="d2", score=0.8, chunk=Chunk(id="c2", document_id="d2", text="java code", start_idx=0, end_idx=9)),
    ]
    
    reranker = CrossEncoderReRanker()
    result = reranker.apply(hits, "python", 1)  # Positional args
    
    assert isinstance(result, TechniqueResult)
    assert result.success is True
    assert len(result.payload["hits"]) == 1
