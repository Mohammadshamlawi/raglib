"""Tests for BM25 (Best Matching 25) retrieval technique."""

from raglib.schemas import Document
from raglib.techniques.bm25 import BM25


def test_bm25_rank_order():
    """Test that BM25 ranks documents by relevance."""
    docs = [
        Document(id="doc1", text="apple fruit red sweet"),
        Document(id="doc2", text="banana fruit yellow"),
        Document(id="doc3", text="apple computer technology"),
        Document(id="doc4", text="fruit vegetables healthy food"),
    ]
    
    bm25 = BM25(docs=docs)
    result = bm25.apply(query="apple fruit")
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) > 0
    
    # Extract document IDs in ranked order
    ranked_docs = [hit.doc_id for hit in hits]
    
    # doc1 should rank highly (has both "apple" and "fruit")
    # doc4 should rank moderately (has "fruit")
    # doc3 should rank moderately (has "apple")
    # doc2 should rank lower (has "fruit" but not "apple")
    
    assert "doc1" in ranked_docs[:2], "doc1 should rank highly"
    assert len(ranked_docs) <= 4


def test_bm25_empty_query_returns_empty_hits():
    """Test that empty query returns empty results."""
    docs = [Document(id="doc1", text="some text")]
    bm25 = BM25(docs=docs)
    
    result = bm25.apply(query="")
    assert result.success
    assert result.payload["hits"] == []


def test_bm25_runtime_corpus():
    """Test that BM25 can accept corpus at query time."""
    bm25 = BM25()  # no initial corpus
    
    docs = [
        Document(id="doc1", text="machine learning algorithms"),
        Document(id="doc2", text="deep learning neural networks"),
    ]
    
    result = bm25.apply(corpus=docs, query="learning", top_k=2)
    
    assert result.success
    hits = result.payload["hits"]
    assert len(hits) == 2
    assert all(hit.score > 0 for hit in hits)


def test_bm25_no_corpus():
    """Test that BM25 handles empty corpus gracefully."""
    bm25 = BM25()
    
    result = bm25.apply(query="test query")
    
    assert result.success
    assert result.payload["hits"] == []
    assert "No documents indexed" in result.meta.get("error", "")
