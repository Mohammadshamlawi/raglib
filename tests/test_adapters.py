from raglib.adapters.dummy_embedder import DummyEmbedder
from raglib.adapters.inmemory_vectorstore import InMemoryVectorStore


def test_dummy_embedder_deterministic():
    embed = DummyEmbedder(dim=8)
    v1 = embed.embed(["hello world"])[0]
    v2 = embed.embed(["hello world"])[0]
    assert isinstance(v1, list)
    assert v1 == v2
    assert len(v1) == 8

def test_inmemory_vectorstore_add_and_search():
    vs = InMemoryVectorStore()
    # very small toy vectors
    ids = ["a", "b"]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    metas = [{"payload": "A"}, {"payload": "B"}]
    vs.add(ids, vectors, metas)
    res = vs.search([1.0, 0.0, 0.0], top_k=2)
    # expect id "a" with highest score
    assert len(res) == 2
    assert res[0][0] == "a"
    assert res[0][2] == {"payload": "A"}
