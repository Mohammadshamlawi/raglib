"""Tests for LLM Generator technique."""

import pytest
from raglib.techniques.llm_generator import LLMGenerator
from raglib.core import TechniqueResult
from raglib.adapters.base import LLMAdapter


class DummyLLMAdapter(LLMAdapter):
    """Test adapter that returns deterministic responses."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        return f"GEN:{prompt}"


def test_generator_with_adapter():
    """Test generator with an LLMAdapter."""
    adapter = DummyLLMAdapter()
    generator = LLMGenerator(llm_adapter=adapter)
    
    result = generator.apply(prompt="hello")
    
    assert isinstance(result, TechniqueResult)
    assert result.success
    assert result.payload["text"] == "GEN:hello"
    assert result.meta["generator"] == "llm_adapter"


def test_generator_with_adapter_multiple_prompts():
    """Test generator with multiple prompts."""
    adapter = DummyLLMAdapter()
    generator = LLMGenerator(llm_adapter=adapter)
    
    result = generator.apply(prompts=["hello", "world"])
    
    assert isinstance(result, TechniqueResult)
    assert result.success
    assert result.payload["texts"] == ["GEN:hello", "GEN:world"]
    assert result.meta["generator"] == "llm_adapter"
    assert result.meta["count"] == 2


def test_generator_with_adapter_and_context():
    """Test generator with context."""
    adapter = DummyLLMAdapter()
    generator = LLMGenerator(llm_adapter=adapter)
    
    result = generator.apply(prompt="hello", context="some context")
    
    assert isinstance(result, TechniqueResult)
    assert result.success
    expected = "GEN:Context: some context\n\nQuery: hello\n\nResponse:"
    assert result.payload["text"] == expected


def test_generator_fallback_deterministic():
    """Test fallback behavior without adapter."""
    generator = LLMGenerator()
    
    # Call twice to ensure deterministic output
    result1 = generator.apply(prompt="hello")
    result2 = generator.apply(prompt="hello")
    
    assert isinstance(result1, TechniqueResult)
    assert result1.success
    assert result1.payload["text"] == "FALLBACK_GENERATION: hello"
    assert result1.meta["generator"] == "fallback"
    
    # Ensure deterministic
    assert result1.payload["text"] == result2.payload["text"]
    assert result1.meta == result2.meta


def test_generator_fallback_multiple_prompts():
    """Test fallback with multiple prompts."""
    generator = LLMGenerator()
    
    result = generator.apply(prompts=["hello", "world"])
    
    assert isinstance(result, TechniqueResult)
    assert result.success
    assert result.payload["texts"] == [
        "FALLBACK_GENERATION: hello",
        "FALLBACK_GENERATION: world"
    ]
    assert result.meta["generator"] == "fallback"


def test_generator_custom_fallback_template():
    """Test generator with custom fallback template."""
    generator = LLMGenerator(fallback_template="CUSTOM: {prompt}")
    
    result = generator.apply(prompt="test")
    
    assert result.payload["text"] == "CUSTOM: test"


def test_generator_with_generation_kwargs():
    """Test generator passes through generation kwargs."""
    
    class TrackingAdapter(LLMAdapter):
        def __init__(self):
            self.last_kwargs = {}
            
        def generate(self, prompt: str, **kwargs) -> str:
            self.last_kwargs = kwargs
            return f"GENERATED:{prompt}"
    
    adapter = TrackingAdapter()
    generator = LLMGenerator(
        llm_adapter=adapter, 
        generation_kwargs={"temperature": 0.7}
    )
    
    generator.apply(prompt="test", max_tokens=100)
    
    # Should include both default and call-specific kwargs
    assert adapter.last_kwargs["temperature"] == 0.7
    assert adapter.last_kwargs["max_tokens"] == 100


def test_generator_error_handling():
    """Test error handling when no prompt provided."""
    generator = LLMGenerator()
    
    result = generator.apply()
    
    assert isinstance(result, TechniqueResult)
    assert not result.success
    assert "Must provide 'prompt' or 'prompts'" in result.payload["error"]


def test_generator_invalid_prompts_type():
    """Test error handling for invalid prompts type."""
    generator = LLMGenerator()
    
    result = generator.apply(prompts="not_a_list")
    
    assert isinstance(result, TechniqueResult)
    assert not result.success
    assert "prompts must be a list" in result.payload["error"]


def test_generator_adapter_error_fallback():
    """Test fallback when adapter raises error."""
    
    class ErrorAdapter(LLMAdapter):
        def generate(self, prompt: str, **kwargs) -> str:
            raise ValueError("Adapter error")
    
    generator = LLMGenerator(llm_adapter=ErrorAdapter())
    
    result = generator.apply(prompt="test")
    
    assert result.success
    assert "ADAPTER_ERROR" in result.payload["text"]
    assert "FALLBACK_GENERATION: test" in result.payload["text"]
