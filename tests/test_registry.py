from raglib.registry import TechniqueRegistry
from raglib.techniques.demo_fixed_chunker import DemoFixedChunker

def test_demo_chunker_auto_registered():
    registry = TechniqueRegistry.list()
    assert "demo_fixed_chunker" in registry
    klass = registry["demo_fixed_chunker"]
    assert klass is DemoFixedChunker
