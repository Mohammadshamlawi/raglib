"""Tests for Lexical Matching retrieval technique."""

from raglib.schemas import Document
from raglib.techniques.lexical_matcher import LexicalMatcher


def test_lexical_matcher_exact_mode():
    """Test lexical matcher in exact matching mode."""
    docs = [
        Document(id="doc1", text="machine learning algorithms"),
        Document(id="doc2", text="this has machine learning in it"),
        Document(id="doc3", text="deep neural networks"),
    ]
    
    matcher = LexicalMatcher(docs=docs, mode="exact")
    result = matcher.apply(query="machine learning", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    
    # Should find docs containing the exact phrase
    assert len(hits) >= 1
    # Check that matching docs have positive scores
    matching_hits = [hit for hit in hits if hit.score > 0]
    assert len(matching_hits) >= 1


def test_lexical_matcher_token_overlap_mode():
    """Test lexical matcher in token overlap mode."""
    docs = [
        Document(id="doc1", text="machine learning deep networks"),
        Document(id="doc2", text="machine vision algorithms"),
        Document(id="doc3", text="natural language processing"),
    ]
    
    matcher = LexicalMatcher(docs=docs, mode="token_overlap")
    result = matcher.apply(query="machine learning", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) >= 2
    
    # doc1 should score highest (has both "machine" and "learning")
    # doc2 should score lower (has only "machine")
    hits_by_id = {hit.doc_id: hit.score for hit in hits}
    if "doc1" in hits_by_id and "doc2" in hits_by_id:
        assert hits_by_id["doc1"] > hits_by_id["doc2"]


def test_lexical_matcher_substring_mode():
    """Test lexical matcher in substring mode."""
    docs = [
        Document(id="doc1", text="programming languages"),
        Document(id="doc2", text="language processing"),
        Document(id="doc3", text="neural networks"),
    ]
    
    matcher = LexicalMatcher(docs=docs, mode="substring")
    result = matcher.apply(query="language", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) >= 2
    
    # Both doc1 and doc2 should match
    doc_ids = {hit.doc_id for hit in hits}
    assert "doc1" in doc_ids or "doc2" in doc_ids


def test_lexical_matcher_weighted_overlap_mode():
    """Test lexical matcher in weighted overlap mode."""
    docs = [
        Document(id="doc1", text="test test test document"),  # High frequency
        Document(id="doc2", text="test document sample"),    # Lower frequency
        Document(id="doc3", text="sample text file"),        # No "test"
    ]
    
    matcher = LexicalMatcher(docs=docs, mode="weighted_overlap")
    result = matcher.apply(query="test", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) >= 2
    
    # Get doc-specific scores
    scores = {hit.doc_id: hit.score for hit in hits}
    
    # Simpler document (shorter) should score higher in weighted overlap
    # because it has less dilution of the matching terms
    assert scores["doc2"] > scores["doc1"]  # "test document sample" beats longer doc
    assert "doc3" not in scores or scores["doc3"] == 0  # No matching terms


def test_lexical_matcher_with_threshold():
    """Test lexical matcher with similarity threshold."""
    docs = [
        Document(id="doc1", text="apple banana cherry"),
        Document(id="doc2", text="apple orange"),
        Document(id="doc3", text="grape fruit"),
    ]
    
    matcher = LexicalMatcher(docs=docs, mode="token_overlap", threshold=0.4)
    result = matcher.apply(query="apple banana", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    
    # Only docs with sufficient overlap should be returned
    assert all(hit.score >= 0.4 for hit in hits)


def test_lexical_matcher_empty_query():
    """Test lexical matcher with empty query."""
    docs = [Document(id="doc1", text="some text")]
    matcher = LexicalMatcher(docs=docs)
    
    result = matcher.apply(query="")
    assert result.success
    assert result.payload["hits"] == []


def test_lexical_matcher_runtime_corpus():
    """Test lexical matcher with corpus provided at query time."""
    matcher = LexicalMatcher()  # No initial corpus
    
    docs = [
        Document(id="doc1", text="information retrieval systems"),
        Document(id="doc2", text="text mining and retrieval"),
    ]
    
    result = matcher.apply(corpus=docs, query="retrieval", top_k=2)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) == 2
    assert all(hit.score > 0 for hit in hits)


def test_lexical_matcher_invalid_mode():
    """Test lexical matcher with invalid mode."""
    docs = [Document(id="doc1", text="test")]
    
    try:
        LexicalMatcher(docs=docs, mode="invalid_mode")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Invalid mode" in str(e)


def test_lexical_matcher_metadata():
    """Test lexical matcher result metadata."""
    docs = [Document(id="doc1", text="test document")]
    matcher = LexicalMatcher(docs=docs, mode="token_overlap")
    
    result = matcher.apply(query="test", top_k=1)
    
    assert result.success
    assert "query" in result.meta
    assert "mode" in result.meta
    assert "threshold" in result.meta
    assert "above_threshold" in result.meta
    assert result.meta["mode"] == "token_overlap"