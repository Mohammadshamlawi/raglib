"""Tests for TF-IDF retrieval technique."""

from raglib.schemas import Document
from raglib.techniques.tfidf import TfIdf


def test_tfidf_basic_functionality():
    """Test basic TF-IDF functionality."""
    docs = [
        Document(id="doc1", text="machine learning algorithms"),
        Document(id="doc2", text="deep learning neural networks"),
        Document(id="doc3", text="natural language processing"),
    ]
    
    tfidf = TfIdf(docs=docs)
    result = tfidf.apply(query="learning", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) == 2  # Only docs with "learning" should match
    assert all(hit.score > 0 for hit in hits)
    
    # Both docs should contain "learning"
    doc_ids = {hit.doc_id for hit in hits}
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


def test_tfidf_cosine_similarity():
    """Test that TF-IDF uses cosine similarity properly."""
    docs = [
        Document(id="doc1", text="apple apple apple"),  # High TF for "apple"
        Document(id="doc2", text="apple orange"),       # Lower TF for "apple"
        Document(id="doc3", text="banana fruit"),       # No "apple"
    ]
    
    tfidf = TfIdf(docs=docs)
    result = tfidf.apply(query="apple", top_k=3)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) >= 2
    
    # doc1 should score higher than doc2 due to higher TF
    hits_by_id = {hit.doc_id: hit.score for hit in hits}
    if "doc1" in hits_by_id and "doc2" in hits_by_id:
        assert hits_by_id["doc1"] > hits_by_id["doc2"]


def test_tfidf_empty_query():
    """Test TF-IDF with empty query."""
    docs = [Document(id="doc1", text="some text")]
    tfidf = TfIdf(docs=docs)
    
    result = tfidf.apply(query="")
    assert result.success
    assert result.payload["hits"] == []


def test_tfidf_runtime_corpus():
    """Test TF-IDF with corpus provided at query time."""
    tfidf = TfIdf()  # No initial corpus
    
    docs = [
        Document(id="doc1", text="information retrieval systems"),
        Document(id="doc2", text="text mining and retrieval"),
    ]
    
    result = tfidf.apply(corpus=docs, query="retrieval", top_k=2)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) == 2
    assert all(hit.score > 0 for hit in hits)


def test_tfidf_normalization():
    """Test TF-IDF with different normalization options."""
    docs = [
        Document(id="doc1", text="test document one"),
        Document(id="doc2", text="test document two"),
    ]
    
    # Test L2 normalization (default)
    tfidf_l2 = TfIdf(docs=docs, norm="l2")
    result_l2 = tfidf_l2.apply(query="test", top_k=2)
    
    # Test no normalization
    tfidf_none = TfIdf(docs=docs, norm="none")
    result_none = tfidf_none.apply(query="test", top_k=2)
    
    assert result_l2.success and result_none.success
    
    # Both should return results, but scores may differ
    assert len(result_l2.payload["hits"]) == 2
    assert len(result_none.payload["hits"]) == 2


def test_tfidf_no_matching_terms():
    """Test TF-IDF when query has no matching terms."""
    docs = [
        Document(id="doc1", text="apple orange banana"),
        Document(id="doc2", text="cat dog bird"),
    ]
    
    tfidf = TfIdf(docs=docs)
    result = tfidf.apply(query="xyz", top_k=5)
    
    assert result.success
    assert result.payload["hits"] == []
    assert "No query terms in vocabulary" in result.meta.get("error", "")


def test_tfidf_metadata():
    """Test TF-IDF result metadata."""
    docs = [Document(id="doc1", text="test document")]
    tfidf = TfIdf(docs=docs)
    
    result = tfidf.apply(query="test", top_k=1)
    
    assert result.success
    assert "query" in result.meta
    assert "query_terms" in result.meta
    assert "corpus_size" in result.meta
    assert "vocabulary_size" in result.meta
    assert result.meta["corpus_size"] == 1
    assert result.meta["vocabulary_size"] >= 1
