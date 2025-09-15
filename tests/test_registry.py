from raglib.registry import TechniqueRegistry
from raglib.techniques.bm25 import BM25


def test_bm25_auto_registered():
    registry = TechniqueRegistry.list()
    assert "BM25 (Best Matching 25)" in registry
    klass = registry["BM25 (Best Matching 25)"]
    assert klass is BM25
