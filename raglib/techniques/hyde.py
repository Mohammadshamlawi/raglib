"""HyDE (Hypothetical Document Embeddings) Technique.

Generates hypothetical answers to improve retrieval through pseudo-document generation.
"""

from typing import Dict, Any, Optional
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..adapters.base import LLMAdapter, EmbedderAdapter
from ..registry import TechniqueRegistry


@TechniqueRegistry.register
class HyDE(RAGTechnique):
    """Hypothetical Document Embeddings (HyDE) technique.
    
    Generates a pseudo-answer (hypothesis) to a query to improve retrieval.
    The generated hypothesis can be used for embedding-based retrieval instead
    of using the original query directly. Registered as "hyde_generator".
    
    Args:
        llm_adapter: Optional LLMAdapter for hypothesis generation
        embedder: Optional EmbedderAdapter for embedding the hypothesis
        generation_kwargs: Default kwargs passed to LLM generation
        fallback_hypothesis_template: Template for deterministic fallback
    
    Usage:
        # With adapters
        hyde = HyDE(llm_adapter=my_llm, embedder=my_embedder)
        result = hyde.apply(query="What is machine learning?")
        
        # With fallback
        hyde = HyDE()  # Uses deterministic fallback
        result = hyde.apply(query="What is ML?", return_embeddings=True)
    """
    
    meta = TechniqueMeta(
        name="hyde_generator", 
        category="retrieval_enhancement",
        description="Generate hypothetical documents to improve retrieval",
        tags={"type": "hypothesis_generation", "fallback": "deterministic"}
    )
    
    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        embedder: Optional[EmbedderAdapter] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        fallback_hypothesis_template: str = "Hypothesis for '{query}': {short_query}"
    ):
        super().__init__(self.meta)
        
        self.llm_adapter = llm_adapter
        self.embedder = embedder
        self.generation_kwargs = generation_kwargs or {}
        self.fallback_hypothesis_template = fallback_hypothesis_template
    
    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply HyDE hypothesis generation.
        
        Args:
            query: The input query (required)
            return_embeddings: Whether to return embeddings of the hypothesis
            top_k: Optional, for downstream compatibility (not used here)
            
        Returns:
            TechniqueResult with hypothesis text and optional embedding
        """
        # Extract arguments
        query = kwargs.get('query') or (args[0] if args else None)
        return_embeddings = kwargs.get('return_embeddings', False)
        
        if query is None:
            return TechniqueResult(
                success=False,
                payload={"error": "Must provide 'query' argument"},
                meta={"method": "hyde"}
            )
        
        # Generate hypothesis
        if self.llm_adapter:
            try:
                # Create a prompt for hypothesis generation
                hypothesis_prompt = self._create_hypothesis_prompt(query)
                hypothesis = self.llm_adapter.generate(hypothesis_prompt, **self.generation_kwargs)
                generator_method = "llm_adapter"
            except Exception:
                # Fallback on adapter error
                hypothesis = self._generate_fallback_hypothesis(query)
                generator_method = "fallback_due_to_error"
        else:
            # Deterministic fallback behavior
            hypothesis = self._generate_fallback_hypothesis(query)
            generator_method = "fallback"
        
        # Prepare result payload
        payload = {"hypothesis": hypothesis}
        meta_info = {"method": "hyde", "generator": generator_method}
        
        # Generate embeddings if requested
        if return_embeddings:
            if self.embedder:
                try:
                    embeddings = self.embedder.embed([hypothesis])
                    payload["embedding"] = embeddings[0] if embeddings else None
                    meta_info["embedder"] = "provided_adapter"
                except Exception:
                    # Fallback to deterministic embedding
                    payload["embedding"] = self._generate_fallback_embedding(hypothesis)
                    meta_info["embedder"] = "fallback_due_to_error"
            else:
                # Deterministic fallback embedding
                payload["embedding"] = self._generate_fallback_embedding(hypothesis)
                meta_info["embedder"] = "fallback"
        
        return TechniqueResult(
            success=True,
            payload=payload,
            meta=meta_info
        )
    
    def _create_hypothesis_prompt(self, query: str) -> str:
        """Create a prompt for hypothesis generation."""
        return (
            f"Given the question: '{query}'\n"
            f"Write a concise, informative answer that directly addresses this question. "
            f"Focus on providing factual information that would be found in a relevant document.\n\n"
            f"Answer:"
        )
    
    def _generate_fallback_hypothesis(self, query: str) -> str:
        """Generate deterministic fallback hypothesis."""
        short_query = query[:50] + "..." if len(query) > 50 else query
        return self.fallback_hypothesis_template.format(
            query=query,
            short_query=short_query
        )
    
    def _generate_fallback_embedding(self, text: str) -> list:
        """Generate deterministic fallback embedding."""
        # Simple deterministic embedding based on text characteristics
        text_lower = text.lower()
        embedding = []
        
        # Use text length and character frequencies for deterministic values
        embedding.append(len(text) / 1000.0)  # Normalize length
        embedding.append(text_lower.count('a') / max(len(text), 1))
        embedding.append(text_lower.count('e') / max(len(text), 1))
        embedding.append(text_lower.count('i') / max(len(text), 1))
        embedding.append(text_lower.count(' ') / max(len(text), 1))
        
        # Pad to a fixed dimension (e.g., 384 dimensions)
        while len(embedding) < 384:
            # Use position-based deterministic values
            pos = len(embedding)
            char_code = ord(text_lower[pos % len(text_lower)]) if text_lower else 97
            embedding.append((char_code % 256) / 256.0)
        
        return embedding[:384]  # Ensure exactly 384 dimensions


# Register the technique
TechniqueRegistry.register(HyDE)
