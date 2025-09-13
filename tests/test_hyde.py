"""Tests for HyDE technique."""

from raglib.techniques.hyde import HyDE
from raglib.core import TechniqueResult
from raglib.adapters.base import LLMAdapter, EmbedderAdapter


class DummyLLMAdapter(LLMAdapter):
    """Test adapter that returns deterministic responses."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Extract the query from the prompt - look for the original question
        if "Given the question:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if line.strip().startswith("'") and line.strip().endswith("'"):
                    question = line.strip()[1:-1]  # Remove quotes
                    return f"HYP:{question.split()[-1]}"
        return "HYP:Answer:"


class DummyEmbedder(EmbedderAdapter):
    """Test embedder that returns deterministic embeddings."""
    
    def embed(self, texts):
        embeddings = []
        for text in texts:
            # Simple deterministic embedding based on text
            embedding = [len(text) / 100.0, text.count('a') / 10.0, text.count('e') / 10.0]
            # Pad to 5 dimensions for testing
            while len(embedding) < 5:
                embedding.append(0.1 * (len(embedding) + 1))
            embeddings.append(embedding[:5])
        return embeddings


def test_hyde_with_adapter():
    """Test HyDE with LLMAdapter."""
    adapter = DummyLLMAdapter()
    hyde = HyDE(llm_adapter=adapter)
    
    result = hyde.apply(query="What is machine learning?")
    
    assert isinstance(result, TechniqueResult)
    assert result.success
    assert result.payload["hypothesis"] == "HYP:Answer:"
    assert result.meta["generator"] == "llm_adapter"
    assert result.meta["method"] == "hyde"
    assert "embedding" not in result.payload


def test_hyde_with_adapter_and_embeddings():
    """Test HyDE with both LLM and embedder adapters."""
    llm_adapter = DummyLLMAdapter()
    embedder = DummyEmbedder()
    hyde = HyDE(llm_adapter=llm_adapter, embedder=embedder)
    
    result = hyde.apply(query="test query", return_embeddings=True)
    
    assert isinstance(result, TechniqueResult)
    assert result.success
    assert result.payload["hypothesis"] == "HYP:Answer:"
    assert "embedding" in result.payload
    assert isinstance(result.payload["embedding"], list)
    assert len(result.payload["embedding"]) == 5
    assert result.meta["embedder"] == "provided_adapter"


def test_hyde_fallback_deterministic():
    """Test deterministic fallback behavior."""
    hyde = HyDE()
    
    # Call twice to ensure deterministic output
    result1 = hyde.apply(query="test question")
    result2 = hyde.apply(query="test question")
    
    assert isinstance(result1, TechniqueResult)
    assert result1.success
    expected = "Hypothesis for 'test question': test question"
    assert result1.payload["hypothesis"] == expected
    assert result1.meta["generator"] == "fallback"
    
    # Ensure deterministic
    assert result1.payload == result2.payload
    assert result1.meta == result2.meta


def test_hyde_fallback_with_embeddings():
    """Test fallback with embeddings."""
    hyde = HyDE()
    
    result = hyde.apply(query="short", return_embeddings=True)
    
    assert isinstance(result, TechniqueResult)
    assert result.success
    assert "hypothesis" in result.payload
    assert "embedding" in result.payload
    assert isinstance(result.payload["embedding"], list)
    assert len(result.payload["embedding"]) == 384  # Fixed dimension
    assert result.meta["embedder"] == "fallback"
    
    # Test embedding determinism
    result2 = hyde.apply(query="short", return_embeddings=True)
    assert result.payload["embedding"] == result2.payload["embedding"]


def test_hyde_long_query_truncation():
    """Test hypothesis generation for long queries."""
    hyde = HyDE()
    long_query = "This is a very long query that exceeds fifty characters in total length"
    
    result = hyde.apply(query=long_query)
    
    assert result.success
    expected = f"Hypothesis for '{long_query}': This is a very long query that exceeds fifty chara..."
    assert result.payload["hypothesis"] == expected


def test_hyde_custom_fallback_template():
    """Test HyDE with custom fallback template."""
    hyde = HyDE(fallback_hypothesis_template="CUSTOM: {query} -> {short_query}")
    
    result = hyde.apply(query="test")
    
    assert result.payload["hypothesis"] == "CUSTOM: test -> test"


def test_hyde_error_handling():
    """Test error handling when no query provided."""
    hyde = HyDE()
    
    result = hyde.apply()
    
    assert isinstance(result, TechniqueResult)
    assert not result.success
    assert "Must provide 'query' argument" in result.payload["error"]


def test_hyde_adapter_error_fallback():
    """Test fallback when adapter raises error."""
    
    class ErrorAdapter(LLMAdapter):
        def generate(self, prompt: str, **kwargs) -> str:
            raise ValueError("Adapter error")
    
    hyde = HyDE(llm_adapter=ErrorAdapter())
    
    result = hyde.apply(query="test")
    
    assert result.success
    assert result.meta["generator"] == "fallback_due_to_error"
    expected = "Hypothesis for 'test': test"
    assert result.payload["hypothesis"] == expected


def test_hyde_embedder_error_fallback():
    """Test fallback when embedder raises error."""
    
    class ErrorEmbedder(EmbedderAdapter):
        def embed(self, texts):
            raise ValueError("Embedder error")
    
    hyde = HyDE(embedder=ErrorEmbedder())
    
    result = hyde.apply(query="test", return_embeddings=True)
    
    assert result.success
    assert result.meta["embedder"] == "fallback_due_to_error" 
    assert isinstance(result.payload["embedding"], list)
    assert len(result.payload["embedding"]) == 384


def test_hyde_generation_kwargs():
    """Test that generation kwargs are passed through."""
    
    class TrackingAdapter(LLMAdapter):
        def __init__(self):
            self.last_kwargs = {}
            
        def generate(self, prompt: str, **kwargs) -> str:
            self.last_kwargs = kwargs
            return "hypothesis"
    
    adapter = TrackingAdapter()
    hyde = HyDE(llm_adapter=adapter, generation_kwargs={"temperature": 0.5})
    
    hyde.apply(query="test")
    
    assert adapter.last_kwargs["temperature"] == 0.5
