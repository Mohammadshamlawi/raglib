from raglib.core import TechniqueResult
from raglib.schemas import Document
from raglib.techniques.bm25_simple import BM25Simple


def test_bm25_simple_rank_order():
    docs = [
        Document(id="d1", text="apple banana"),
        Document(id="d2", text="banana banana apple"),
        Document(id="d3", text="orange fruit banana"),
    ]
    bm25 = BM25Simple(docs=docs)
    res = bm25.apply(query="banana", top_k=3)
    assert isinstance(res, TechniqueResult)
    hits = res.payload["hits"]
    # Expect doc d2 (with two bananas) to score highest
    assert hits[0].doc_id == "d2"
    # All three docs returned
    assert {h.doc_id for h in hits} == {"d1", "d2", "d3"}


def test_bm25_empty_query_returns_empty_hits():
    bm25 = BM25Simple(docs=[])
    res = bm25.apply(query="", top_k=5)
    assert res.payload["hits"] == []
