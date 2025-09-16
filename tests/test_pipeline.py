from raglib.pipelines import Pipeline
from raglib.schemas import Document
from raglib.techniques.fixed_size_chunker import FixedSizeChunker
from raglib.techniques.semantic_chunker import SemanticChunker


def test_pipeline_chunker_basic_functionality():
    text = "This is a test document. " * 10  # longer text
    doc = Document(id="pdoc1", text=text)
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)  # valid overlap
    pipeline = Pipeline([chunker])
    result = pipeline.run(doc)  # default return_payload_only=True
    assert isinstance(result, dict)
    assert "chunks" in result
    chunks = result["chunks"]
    assert len(chunks) >= 1
    assert all(hasattr(c, "text") for c in chunks)


def test_pipeline_multiple_chunkers():
    text = "This is a test document for pipeline testing. " * 5
    doc = Document(id="pdoc2", text=text)
    chunker1 = FixedSizeChunker(chunk_size=100)
    chunker2 = SemanticChunker()
    pipeline = Pipeline([chunker1, chunker2])
    # Just test that pipeline runs without error
    pipeline.run(doc)
