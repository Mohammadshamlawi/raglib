from raglib.adapters.dummy_embedder import DummyEmbedder
from raglib.adapters.inmemory_vectorstore import InMemoryVectorStore
from raglib.techniques.dummy_dense import InMemoryDenseRetriever
from raglib.schemas import Chunk
from raglib.core import TechniqueResult

def test_dummy_dense_retrieval_basic_with_adapters():
    chunks = [
        Chunk(id="c1", document_id="docA", text="cat dog", start_idx=0, end_idx=7),
        Chunk(id="c2", document_id="docB", text="elephant", start_idx=0, end_idx=8),
        Chunk(id="c3", document_id="docC", text="dog dog cat", start_idx=0, end_idx=12),
    ]
    embedder = DummyEmbedder(dim=16)
    vs = InMemoryVectorStore()
    retriever = InMemoryDenseRetriever(embedder=embedder, vectorstore=vs, chunks=chunks)
    res = retriever.apply(query="dog", top_k=2)
    assert isinstance(res, TechniqueResult)
    hits = res.payload["hits"]
    assert len(hits) == 2
    assert hits[0].chunk is not None
    assert hits[0].chunk.id == "c3"
    assert hits[0].doc_id == "docC"

def test_dummy_dense_no_chunks_returns_empty_with_adapters():
    embedder = DummyEmbedder(dim=8)
    vs = InMemoryVectorStore()
    retriever = InMemoryDenseRetriever(embedder=embedder, vectorstore=vs, chunks=None)
    res = retriever.apply(query="anything", top_k=5)
    assert res.payload["hits"] == []
