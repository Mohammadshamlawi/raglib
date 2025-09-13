"""LLM Generator Technique.

Wrapper for LLMAdapter to provide text generation capabilities with fallback behavior.
"""

from typing import Dict, Any, Optional
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..adapters.base import LLMAdapter
from ..registry import TechniqueRegistry


@TechniqueRegistry.register
class LLMGenerator(RAGTechnique):
    """LLM text generation technique.
    
    Wraps an LLMAdapter to generate text responses with deterministic fallback
    when no adapter is provided. Registered as "llm_generator".
    
    Args:
        llm_adapter: Optional LLMAdapter for text generation
        generation_kwargs: Default kwargs passed to adapter's generate method
        fallback_template: Template for fallback responses (default: "{prompt}")
    
    Usage:
        # With adapter
        generator = LLMGenerator(llm_adapter=my_llm)
        result = generator.apply(prompt="Hello world")
        
        # With fallback
        generator = LLMGenerator()  # Uses deterministic fallback
        result = generator.apply(prompt="Hello world")
        
        # Multiple prompts
        result = generator.apply(prompts=["Hello", "World"])
    """
    
    meta = TechniqueMeta(
        name="llm_generator",
        category="generation",
        description="LLM text generation with deterministic fallback",
        tags={"type": "generator", "fallback": "deterministic"}
    )
    
    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        fallback_template: str = "FALLBACK_GENERATION: {prompt}"
    ):
        super().__init__(self.meta)
        
        self.llm_adapter = llm_adapter
        self.generation_kwargs = generation_kwargs or {}
        self.fallback_template = fallback_template
    
    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply text generation.
        
        Accepts:
        - apply(prompt: str, **kwargs) — single prompt generation
        - apply(prompts: List[str], **kwargs) — multiple prompt generation
        - apply(prompt: str, context: Optional[str] = None, **kwargs) — with context
        
        Returns:
            TechniqueResult with payload containing "text" (single) or "texts" (multiple)
        """
        # Extract arguments
        prompt = kwargs.get('prompt') or (args[0] if args else None)
        prompts = kwargs.get('prompts')
        context = kwargs.get('context')
        
        # Override generation kwargs for this call
        call_generation_kwargs = {**self.generation_kwargs}
        call_generation_kwargs.update({k: v for k, v in kwargs.items() 
                                     if k not in ['prompt', 'prompts', 'context']})
        
        # Handle multiple prompts case
        if prompts:
            if not isinstance(prompts, list):
                return TechniqueResult(
                    success=False,
                    payload={"error": "prompts must be a list"},
                    meta={"generator": "error"}
                )
            
            texts = []
            for p in prompts:
                final_prompt = self._build_prompt(p, context)
                text = self._generate_single(final_prompt, call_generation_kwargs)
                texts.append(text)
            
            generator_type = "llm_adapter" if self.llm_adapter else "fallback"
            return TechniqueResult(
                success=True,
                payload={"texts": texts},
                meta={"generator": generator_type, "count": len(texts)}
            )
        
        # Handle single prompt case
        if prompt is None:
            return TechniqueResult(
                success=False,
                payload={"error": "Must provide 'prompt' or 'prompts'"},
                meta={"generator": "error"}
            )
        
        final_prompt = self._build_prompt(prompt, context)
        text = self._generate_single(final_prompt, call_generation_kwargs)
        
        generator_type = "llm_adapter" if self.llm_adapter else "fallback"
        return TechniqueResult(
            success=True,
            payload={"text": text},
            meta={"generator": generator_type}
        )
    
    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Build the final prompt including optional context."""
        if context:
            return f"Context: {context}\n\nQuery: {prompt}\n\nResponse:"
        return prompt
    
    def _generate_single(self, prompt: str, generation_kwargs: Dict[str, Any]) -> str:
        """Generate text for a single prompt."""
        if self.llm_adapter:
            try:
                return self.llm_adapter.generate(prompt, **generation_kwargs)
            except Exception as e:
                # Fallback on adapter error
                return f"ADAPTER_ERROR: {str(e)} | {self.fallback_template.format(prompt=prompt)}"
        else:
            # Deterministic fallback behavior
            return self.fallback_template.format(prompt=prompt)
