import pytest

from raglib.core import RAGTechnique, TechniqueMeta, TechniqueResult
from raglib.registry import TechniqueRegistry


def test_rag_abstract_cannot_instantiate():
    # RAGTechnique is abstract; attempting to instantiate should raise a TypeError
    with pytest.raises(TypeError):
        RAGTechnique()  # abstract, missing apply

def test_register_and_registry_listing():
    @TechniqueRegistry.register
    class DummyTechnique(RAGTechnique):
        meta = TechniqueMeta(name="dummy", category="test", description="A dummy technique for tests")

        def __init__(self):
            super().__init__(self.meta)

        def apply(self, *args, **kwargs):
            return TechniqueResult(success=True, payload={"echo": {"args": args, "kwargs": kwargs}})

    # the registry should include the dummy technique
    registry = TechniqueRegistry.list()
    assert "dummy" in registry
    klass = registry["dummy"]
    instance = klass()
    result = instance.apply(1, a=2)
    assert isinstance(result, TechniqueResult)
    assert result.success is True
    assert "echo" in result.payload
