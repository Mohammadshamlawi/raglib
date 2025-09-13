import pytest
from raglib.pipelines import Pipeline
from raglib.techniques.demo_fixed_chunker import DemoFixedChunker
from raglib.techniques.echo_technique import EchoTechnique
from raglib.techniques.null_technique import NullTechnique
from raglib.schemas import Document

def test_pipeline_chunker_then_echo_returns_chunks():
    text = "abcdefghijklmnopqrstuvwxyz" * 5  # 130 chars
    doc = Document(id="pdoc1", text=text)
    chunker = DemoFixedChunker()
    echo = EchoTechnique()
    pipeline = Pipeline([chunker, echo])
    result = pipeline.run(doc)  # default return_payload_only=True
    # echo returns the chunker payload (a dict with "chunks")
    assert isinstance(result, dict)
    assert "chunks" in result
    chunks = result["chunks"]
    assert len(chunks) >= 1
    assert all(c.document_id == "pdoc1" for c in chunks)

def test_pipeline_strict_mode_raises_on_null_step():
    text = "short text"
    doc = Document(id="pdoc2", text=text)
    chunker = DemoFixedChunker()
    null_step = NullTechnique()
    pipeline = Pipeline([chunker, null_step])
    with pytest.raises(RuntimeError):
        pipeline.run(doc, strict=True)
