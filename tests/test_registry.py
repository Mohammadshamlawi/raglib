from raglib.registry import TechniqueRegistry
from raglib.techniques.bm25_simple import BM25Simple


def test_bm25_simple_auto_registered():
    registry = TechniqueRegistry.list()
    assert "bm25_simple" in registry
    klass = registry["bm25_simple"]
    assert klass is BM25Simple
