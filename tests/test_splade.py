"""Tests for SPLADE retrieval technique."""

from raglib.schemas import Document
from raglib.techniques.splade import Splade


def test_splade_basic_functionality():
    """Test basic SPLADE functionality."""
    docs = [
        Document(id="doc1", text="machine learning algorithms"),
        Document(id="doc2", text="deep learning neural networks"),
        Document(id="doc3", text="natural language processing"),
    ]
    
    splade = Splade(docs=docs)
    result = splade.apply(query="learning", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) >= 2  # Should find docs with "learning"
    assert all(hit.score > 0 for hit in hits)


def test_splade_term_expansion():
    """Test SPLADE term expansion functionality."""
    docs = [
        Document(id="doc1", text="machine learning algorithms"),
        Document(id="doc2", text="algorithmic approaches"),
        Document(id="doc3", text="natural language processing"),
    ]
    
    splade = Splade(docs=docs, expansion_factor=0.5)
    result = splade.apply(query="algorithm", top_k=3)
    
    assert result.success
    assert "expanded_terms" in result.meta
    
    # Should include original and expanded terms
    expanded_terms = result.meta["expanded_terms"]
    assert "algorithm" in expanded_terms


def test_splade_sparsity_control():
    """Test SPLADE sparsity threshold control."""
    docs = [
        Document(id="doc1", text="test document with many words"),
        Document(id="doc2", text="another test document"),
    ]
    
    # High sparsity threshold should reduce term count
    splade_sparse = Splade(docs=docs, sparsity_threshold=0.1)
    result_sparse = splade_sparse.apply(query="test", top_k=2)
    
    # Low sparsity threshold should include more terms
    splade_dense = Splade(docs=docs, sparsity_threshold=0.001)
    result_dense = splade_dense.apply(query="test", top_k=2)
    
    assert result_sparse.success and result_dense.success
    
    # Both should return results
    assert len(result_sparse.payload["hits"]) > 0
    assert len(result_dense.payload["hits"]) > 0


def test_splade_expansion_factor():
    """Test SPLADE expansion factor effects."""
    docs = [
        Document(id="doc1", text="programming languages"),
        Document(id="doc2", text="programming paradigms"),
    ]
    
    # Test with different expansion factors
    splade_low = Splade(docs=docs, expansion_factor=0.1)
    splade_high = Splade(docs=docs, expansion_factor=0.8)
    
    result_low = splade_low.apply(query="program", top_k=2)
    result_high = splade_high.apply(query="program", top_k=2)
    
    assert result_low.success and result_high.success
    
    # Both should find relevant documents
    assert len(result_low.payload["hits"]) > 0
    assert len(result_high.payload["hits"]) > 0


def test_splade_empty_query():
    """Test SPLADE with empty query."""
    docs = [Document(id="doc1", text="some text")]
    splade = Splade(docs=docs)
    
    result = splade.apply(query="")
    assert result.success
    assert result.payload["hits"] == []


def test_splade_runtime_corpus():
    """Test SPLADE with corpus provided at query time."""
    splade = Splade()  # No initial corpus
    
    docs = [
        Document(id="doc1", text="information retrieval systems"),
        Document(id="doc2", text="text mining and retrieval"),
    ]
    
    result = splade.apply(corpus=docs, query="retrieval", top_k=2)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) == 2
    assert all(hit.score > 0 for hit in hits)


def test_splade_no_valid_terms():
    """Test SPLADE when query has no valid terms."""
    docs = [
        Document(id="doc1", text="apple orange banana"),
        Document(id="doc2", text="cat dog bird"),
    ]
    
    splade = Splade(docs=docs)
    result = splade.apply(query="xyz", top_k=5)
    
    assert result.success
    # Should still try to expand, but may not find matches
    assert isinstance(result.payload["hits"], list)


def test_splade_scoring():
    """Test SPLADE scoring produces reasonable results."""
    docs = [
        # 2 'learning' out of 5 tokens
        Document(id="doc1", text="machine learning deep learning models"),
        # 1 'learning' out of 2 tokens
        Document(id="doc2", text="machine learning"),
        # No 'learning'
        Document(id="doc3", text="natural processing"),
    ]
    
    splade = Splade(docs=docs)
    result = splade.apply(query="learning", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    
    # Should find documents with "learning"
    relevant_hits = [hit for hit in hits if hit.score > 0]
    assert len(relevant_hits) >= 2
    
    # doc2 should score higher than doc1 (higher relative frequency: 1/2 > 2/5)
    hits_by_id = {hit.doc_id: hit.score for hit in hits}
    if "doc1" in hits_by_id and "doc2" in hits_by_id:
        assert hits_by_id["doc2"] > hits_by_id["doc1"]


def test_splade_metadata():
    """Test SPLADE result metadata."""
    docs = [Document(id="doc1", text="test document")]
    splade = Splade(docs=docs)
    
    result = splade.apply(query="test", top_k=1)
    
    assert result.success
    assert "query" in result.meta
    assert "expanded_terms" in result.meta
    assert "vocabulary_size" in result.meta
    assert "expansion_factor" in result.meta
    assert result.meta["expansion_factor"] == 0.3  # default value