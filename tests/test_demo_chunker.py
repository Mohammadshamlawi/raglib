from raglib.techniques.demo_fixed_chunker import DemoFixedChunker
from raglib.schemas import Document
from raglib.core import TechniqueResult

def test_demo_fixed_chunker_splits_text():
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars
    doc = Document(id="testdoc1", text=text)
    chunker = DemoFixedChunker()
    result = chunker.apply(doc, chunk_size=100, overlap=10)
    assert isinstance(result, TechniqueResult)
    assert result.success is True
    payload = result.payload
    assert "chunks" in payload
    chunks = payload["chunks"]
    # check sizes and coverage
    assert len(chunks) >= 3
    # each chunk text length <= chunk_size
    assert all(len(c.text) <= 100 for c in chunks)
    # check chunk ids and document_id correctness
    assert all(c.document_id == "testdoc1" for c in chunks)
    assert all(c.id.startswith("testdoc1_chunk_") for c in chunks)
