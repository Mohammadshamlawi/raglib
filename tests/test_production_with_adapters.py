from raglib.adapters.dummy_embedder import DummyEmbedder
from raglib.adapters.inmemory_vectorstore import InMemoryVectorStore
from raglib.techniques.dense_retriever import DenseRetriever
from raglib.schemas import Chunk

def test_dense_retriever_with_provided_adapters_matches_default_behaviour():
    chunks = [
        Chunk(id="c1", document_id="d1", text="dog cat", start_idx=0, end_idx=7),
        Chunk(id="c2", document_id="d2", text="dog dog dog", start_idx=0, end_idx=11),
        Chunk(id="c3", document_id="d3", text="elephant", start_idx=0, end_idx=8),
    ]
    # default retriever (uses internal DummyEmbedder + InMemoryVectorStore)
    default_retriever = DenseRetriever(chunks=chunks)
    res_default = default_retriever.apply(query="dog", top_k=2)
    top_default_id = res_default.payload["hits"][0].chunk.id

    # explicit adapter-backed retriever
    embedder = DummyEmbedder(dim=16)
    vs = InMemoryVectorStore()
    adapter_retriever = DenseRetriever(embedder=embedder, vectorstore=vs, chunks=chunks)
    res_adapter = adapter_retriever.apply(query="dog", top_k=2)
    top_adapter_id = res_adapter.payload["hits"][0].chunk.id

    # deterministic embedder should produce same top result
    assert top_default_id == top_adapter_id
