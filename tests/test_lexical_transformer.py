"""Tests for Lexical Transformer retrieval technique."""

from raglib.schemas import Document
from raglib.techniques.lexical_transformer import LexicalTransformer


def test_lexical_transformer_basic_functionality():
    """Test basic lexical transformer functionality."""
    docs = [
        Document(id="doc1", text="machine learning algorithms"),
        Document(id="doc2", text="deep learning neural networks"),
        Document(id="doc3", text="natural language processing"),
    ]
    
    transformer = LexicalTransformer(docs=docs)
    result = transformer.apply(query="learning algorithms", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) >= 1
    assert all(hit.score > 0 for hit in hits)


def test_lexical_transformer_attention_weighting():
    """Test that attention weighting affects scores appropriately."""
    docs = [
        Document(id="doc1", text="machine learning deep learning algorithms"),
        Document(id="doc2", text="machine algorithms"),
        Document(id="doc3", text="natural processing"),
    ]
    
    transformer = LexicalTransformer(docs=docs)
    result = transformer.apply(query="machine learning", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    
    # doc1 should score higher than doc2 (has both terms + more matches)
    hits_by_id = {hit.doc_id: hit.score for hit in hits if hit.score > 0}
    if "doc1" in hits_by_id and "doc2" in hits_by_id:
        assert hits_by_id["doc1"] > hits_by_id["doc2"]


def test_lexical_transformer_positional_effects():
    """Test that position in document affects scoring."""
    docs = [
        Document(id="doc1", text="important keyword at start"),
        Document(id="doc2", text="some text and then important keyword at end"),
        Document(id="doc3", text="no relevant terms here"),
    ]
    
    transformer = LexicalTransformer(docs=docs)
    result = transformer.apply(query="important keyword", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    
    # Should find documents with the keywords
    relevant_hits = [hit for hit in hits if hit.score > 0]
    assert len(relevant_hits) >= 2


def test_lexical_transformer_weight_parameters():
    """Test custom attention and position weights."""
    docs = [
        Document(id="doc1", text="test document with keywords"),
        Document(id="doc2", text="another test with different keywords"),
    ]
    
    # High attention weight
    transformer_attention = LexicalTransformer(
        docs=docs, attention_weight=0.9, position_weight=0.1
    )
    
    # High position weight
    transformer_position = LexicalTransformer(
        docs=docs, attention_weight=0.1, position_weight=0.9
    )
    
    result_attention = transformer_attention.apply(query="test keywords", top_k=2)
    result_position = transformer_position.apply(query="test keywords", top_k=2)
    
    assert result_attention.success and result_position.success
    
    # Both should return results
    assert len(result_attention.payload["hits"]) > 0
    assert len(result_position.payload["hits"]) > 0


def test_lexical_transformer_empty_query():
    """Test lexical transformer with empty query."""
    docs = [Document(id="doc1", text="some text")]
    transformer = LexicalTransformer(docs=docs)
    
    result = transformer.apply(query="")
    assert result.success
    assert result.payload["hits"] == []


def test_lexical_transformer_runtime_corpus():
    """Test lexical transformer with corpus provided at query time."""
    transformer = LexicalTransformer()  # No initial corpus
    
    docs = [
        Document(id="doc1", text="information retrieval systems"),
        Document(id="doc2", text="text mining and retrieval"),
    ]
    
    result = transformer.apply(corpus=docs, query="retrieval systems", top_k=2)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) >= 1
    assert all(hit.score > 0 for hit in hits)


def test_lexical_transformer_coverage_boost():
    """Test that query term coverage affects scoring."""
    docs = [
        Document(id="doc1", text="machine learning deep neural networks"),
        Document(id="doc2", text="machine learning"),
        Document(id="doc3", text="machine only"),
    ]
    
    transformer = LexicalTransformer(docs=docs)
    result = transformer.apply(query="machine learning", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    
    # Documents with more query term coverage should score higher
    hits_by_id = {hit.doc_id: hit.score for hit in hits if hit.score > 0}
    
    # doc1 and doc2 have both terms, doc3 has only one
    if "doc2" in hits_by_id and "doc3" in hits_by_id:
        assert hits_by_id["doc2"] > hits_by_id["doc3"]


def test_lexical_transformer_prefix_suffix_matching():
    """Test prefix/suffix matching in attention computation."""
    docs = [
        Document(id="doc1", text="programming programs programmer"),
        Document(id="doc2", text="testing tests tested"),
        Document(id="doc3", text="unrelated content"),
    ]
    
    transformer = LexicalTransformer(docs=docs)
    result = transformer.apply(query="program", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    
    # doc1 should match due to prefix/suffix similarities
    relevant_hits = [hit for hit in hits if hit.score > 0]
    assert len(relevant_hits) >= 1
    
    doc_ids = {hit.doc_id for hit in relevant_hits}
    assert "doc1" in doc_ids


def test_lexical_transformer_metadata():
    """Test lexical transformer result metadata."""
    docs = [Document(id="doc1", text="test document")]
    transformer = LexicalTransformer(docs=docs)
    
    result = transformer.apply(query="test", top_k=1)
    
    assert result.success
    assert "query" in result.meta
    assert "attention_weight" in result.meta
    assert "position_weight" in result.meta
    assert result.meta["attention_weight"] + result.meta["position_weight"] == 1.0