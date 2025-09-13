from raglib.techniques.bm25_production import BM25Retriever
from raglib.techniques.dense_retriever import DenseRetriever
from raglib.schemas import Document, Chunk
from raglib.core import TechniqueResult

def test_bm25_retriever_works_with_corpus_on_apply():
    docs = [
        Document(id="p1", text="apple orange"),
        Document(id="p2", text="banana banana apple"),
        Document(id="p3", text="banana fruit"),
    ]
    retriever = BM25Retriever()  # no docs pre-indexed
    res = retriever.apply(corpus=docs, query="banana", top_k=3)
    assert isinstance(res, TechniqueResult)
    hits = res.payload["hits"]
    assert hits[0].doc_id == "p2"
    assert len(hits) == 3

def test_dense_retriever_default_fallback_works_without_adapters():
    chunks = [
        Chunk(id="c1", document_id="d1", text="dog cat", start_idx=0, end_idx=7),
        Chunk(id="c2", document_id="d2", text="dog dog dog", start_idx=0, end_idx=11),
        Chunk(id="c3", document_id="d3", text="elephant", start_idx=0, end_idx=8),
    ]
    # Construct without adapters; class will fall back to dummy embedder and in-memory VS
    retriever = DenseRetriever(chunks=chunks)
    # query
    res = retriever.apply(query="dog", top_k=2)
    assert isinstance(res, TechniqueResult)
    hits = res.payload["hits"]
    assert len(hits) == 2
    assert hits[0].chunk is not None
    # top chunk should be the repeated-dog chunk (c2)
    assert hits[0].chunk.id == "c2"
