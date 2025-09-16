from raglib.registry import TechniqueRegistry
from raglib.techniques.bm25 import BM25


def test_bm25_auto_registered():
    registry = TechniqueRegistry.list()
    assert "bm25" in registry
    klass = registry["bm25"]
    assert klass is BM25
