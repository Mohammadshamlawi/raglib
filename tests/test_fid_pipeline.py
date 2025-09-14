"""Tests for Fusion-in-Decoder Pipeline architecture."""

import pytest

from raglib.adapters.base import LLMAdapter
from raglib.architectures.fid import FusionInDecoderPipeline
from raglib.core import RAGTechnique, TechniqueMeta, TechniqueResult
from raglib.schemas import Chunk, Hit


class DummyRetriever(RAGTechnique):
    """Test retriever that returns fixed hits."""

    def __init__(self, hits=None):
        meta = TechniqueMeta(
            name="dummy_retriever",
            category="retrieval",
            description="Test retriever"
        )
        super().__init__(meta)
        self.hits = hits or []

    def apply(self, *args, **kwargs):
        top_k = kwargs.get('top_k', 5)
        return TechniqueResult(
            success=True,
            payload={"hits": self.hits[:top_k]},
            meta={"retrieved": len(self.hits[:top_k])}
        )


class DummyLLMAdapter(LLMAdapter):
    """Test adapter that returns deterministic responses."""

    def generate(self, prompt: str, **kwargs) -> str:
        return f"ANSWER_FOR: {prompt.split('Query:')[1].split('Context:')[0].strip() if 'Query:' in prompt else 'unknown'}"


class DummyGenerator(RAGTechnique):
    """Test generator technique."""

    def __init__(self):
        meta = TechniqueMeta(
            name="dummy_generator",
            category="generation",
            description="Test generator"
        )
        super().__init__(meta)

    def apply(self, *args, **kwargs):
        prompt = kwargs.get('prompt', '')
        return TechniqueResult(
            success=True,
            payload={"text": f"GENERATED: {prompt}"},
            meta={"generator": "dummy"}
        )


def test_fid_pipeline_separate_mode():
    """Test FiD pipeline in separate mode."""
    # Create test hits
    chunk1 = Chunk(id="c1", document_id="doc1", text="Machine learning is AI", start_idx=0, end_idx=22)
    chunk2 = Chunk(id="c2", document_id="doc2", text="Deep learning uses neural nets", start_idx=0, end_idx=30)

    hits = [
        Hit(doc_id="doc1", score=0.9, chunk=chunk1),
        Hit(doc_id="doc2", score=0.8, chunk=chunk2)
    ]

    retriever = DummyRetriever(hits=hits)
    generator = DummyGenerator()

    pipeline = FusionInDecoderPipeline(
        retriever=retriever,
        generator=generator,
        mode="separate",
        top_k=2
    )

    result = pipeline.run(query="What is machine learning?")

    assert isinstance(result, TechniqueResult)
    assert result.success
    assert "answer" in result.payload
    assert "component_outputs" in result.payload
    assert "hits" in result.payload

    # Should have two component outputs (one per context)
    assert len(result.payload["component_outputs"]) == 2

    # Check that answer contains both generated parts
    answer = result.payload["answer"]
    assert "GENERATED:" in answer
    assert "---" in answer  # Separator between components

    assert result.meta["method"] == "fid"
    assert result.meta["mode"] == "separate"
    assert result.meta["contexts_processed"] == 2


def test_fid_pipeline_concat_mode():
    """Test FiD pipeline in concat mode."""
    chunk1 = Chunk(id="c1", document_id="doc1", text="Python is great", start_idx=0, end_idx=15)
    hits = [Hit(doc_id="doc1", score=0.9, chunk=chunk1)]

    retriever = DummyRetriever(hits=hits)
    generator = DummyGenerator()

    pipeline = FusionInDecoderPipeline(
        retriever=retriever,
        generator=generator,
        mode="concat",
        top_k=1
    )

    result = pipeline.run(query="Tell me about Python")

    assert result.success
    assert len(result.payload["component_outputs"]) == 1  # Single concatenated call

    component = result.payload["component_outputs"][0]
    assert component["contexts_count"] == 1
    assert "Context 1:" in component["prompt"]


def test_fid_pipeline_with_llm_adapter():
    """Test FiD pipeline using LLMAdapter directly."""
    chunk1 = Chunk(id="c1", document_id="doc1", text="Testing with adapter", start_idx=0, end_idx=20)
    hits = [Hit(doc_id="doc1", score=1.0, chunk=chunk1)]

    retriever = DummyRetriever(hits=hits)
    adapter = DummyLLMAdapter()

    pipeline = FusionInDecoderPipeline(
        retriever=retriever,
        llm_adapter=adapter,
        mode="separate"
    )

    result = pipeline.run(query="test query")

    assert result.success
    assert "ANSWER_FOR: test query" in result.payload["answer"]


def test_fid_pipeline_requires_generator_or_adapter():
    """Test FiD pipeline fails when no generator or adapter provided."""
    retriever = DummyRetriever(hits=[])

    # No generator or adapter provided - should raise ValueError
    with pytest.raises(ValueError, match="requires either a 'generator' technique or an 'llm_adapter'"):
        FusionInDecoderPipeline(retriever=retriever)


def test_fid_pipeline_no_hits():
    """Test FiD pipeline when retriever returns no hits."""
    retriever = DummyRetriever(hits=[])  # Empty hits
    generator = DummyGenerator()

    pipeline = FusionInDecoderPipeline(retriever=retriever, generator=generator)

    result = pipeline.run(query="no results")

    assert result.success
    assert result.payload["answer"] == "No relevant documents found."
    assert result.payload["component_outputs"] == []
    assert result.payload["hits"] == []


def test_fid_pipeline_retrieval_failure():
    """Test FiD pipeline when retrieval fails."""

    class FailingRetriever(RAGTechnique):
        def __init__(self):
            meta = TechniqueMeta(name="failing", category="retrieval", description="Fails")
            super().__init__(meta)

        def apply(self, *args, **kwargs):
            return TechniqueResult(success=False, payload={"error": "Retrieval failed"})

    pipeline = FusionInDecoderPipeline(
        retriever=FailingRetriever(),
        generator=DummyGenerator()
    )

    result = pipeline.run(query="test")

    assert not result.success
    assert "Retrieval failed" in result.payload["error"]


def test_fid_pipeline_different_hit_formats():
    """Test FiD pipeline with different hit text formats."""
    # Hit with text attribute directly
    hit1 = Hit(doc_id="doc1", score=1.0)
    hit1.text = "Direct text attribute"

    # Hit with meta text
    hit2 = Hit(doc_id="doc2", score=0.9, meta={"text": "Text in meta"})

    # Hit with chunk
    chunk = Chunk(id="c3", document_id="doc3", text="Text in chunk", start_idx=0, end_idx=13)
    hit3 = Hit(doc_id="doc3", score=0.8, chunk=chunk)

    retriever = DummyRetriever(hits=[hit1, hit2, hit3])
    generator = DummyGenerator()

    pipeline = FusionInDecoderPipeline(
        retriever=retriever,
        generator=generator,
        mode="separate"
    )

    result = pipeline.run(query="test different formats")

    assert result.success
    assert len(result.payload["component_outputs"]) == 3


def test_fid_pipeline_top_k_override():
    """Test FiD pipeline with top_k override in run method."""
    hits = [
        Hit(doc_id=f"doc{i}", score=1.0-i*0.1, chunk=Chunk(
            id=f"c{i}", document_id=f"doc{i}", text=f"Text {i}",
            start_idx=0, end_idx=6
        ))
        for i in range(5)
    ]

    retriever = DummyRetriever(hits=hits)
    generator = DummyGenerator()

    pipeline = FusionInDecoderPipeline(
        retriever=retriever,
        generator=generator,
        top_k=5  # Default top_k
    )

    # Override with top_k=2
    result = pipeline.run(query="test", top_k=2)

    assert result.success
    assert len(result.payload["component_outputs"]) == 2
    assert result.meta["contexts_processed"] == 2


def test_fid_pipeline_unknown_mode():
    """Test FiD pipeline with unknown mode."""
    hits = [Hit(doc_id="doc1", score=1.0, chunk=Chunk(
        id="c1", document_id="doc1", text="test", start_idx=0, end_idx=4
    ))]

    pipeline = FusionInDecoderPipeline(
        retriever=DummyRetriever(hits=hits),
        generator=DummyGenerator(),
        mode="unknown_mode"
    )

    result = pipeline.run(query="test")

    assert not result.success
    assert "Unknown mode: unknown_mode" in result.payload["error"]
