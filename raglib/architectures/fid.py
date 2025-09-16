"""Fusion-in-Decoder Pipeline Architecture.

This module provides a FiD-style orchestrator that combines retrieval and generation
in a way that processes multiple retrieved contexts either separately or concatenated.
"""

from typing import Dict, Optional

from ..adapters.base import LLMAdapter
from ..core import RAGTechnique, TechniqueResult


class FusionInDecoderPipeline:
    """Fusion-in-Decoder style pipeline orchestrator.
    
    This orchestrator:
    1. Uses a retriever technique to get top-K documents for a query
    2. Processes retrieved contexts with a generator technique
    3. Returns a final aggregated answer
    
    Args:
        retriever: Any RAGTechnique that returns results with "hits" payload
        generator: Optional RAGTechnique for text generation (if None, uses llm_adapter)
        llm_adapter: Optional LLMAdapter (alternative to generator)
        top_k: Number of documents to retrieve (default: 5)
        mode: Processing mode - "separate" (per-document) or "concat" (all at once)
        generator_kwargs: Optional kwargs passed to generator
    """

    def __init__(
        self,
        retriever: RAGTechnique,
        generator: Optional[RAGTechnique] = None,
        llm_adapter: Optional[LLMAdapter] = None,
        top_k: int = 5,
        mode: str = "separate",
        generator_kwargs: Optional[Dict] = None
    ):
        self.retriever = retriever
        self.generator = generator
        self.llm_adapter = llm_adapter
        self.top_k = top_k
        self.mode = mode
        self.generator_kwargs = generator_kwargs or {}

        # Require either generator or llm_adapter to be provided
        if not self.generator and not self.llm_adapter:
            raise ValueError(
                "FusionInDecoderPipeline requires either a 'generator' technique "
                "or an 'llm_adapter' to be provided"
            )

    def run(self, query: str, *, top_k: Optional[int] = None, **kwargs) -> TechniqueResult:
        """Run the FiD pipeline.
        
        Args:
            query: The input query
            top_k: Override default top_k for this run
            **kwargs: Additional arguments passed to retriever
            
        Returns:
            TechniqueResult with answer, component_outputs, and hits
        """
        # Step 1: Retrieve documents
        retrieval_top_k = top_k or self.top_k
        retrieval_result = self.retriever.apply(query=query, top_k=retrieval_top_k, **kwargs)

        if not retrieval_result.success or "hits" not in retrieval_result.payload:
            return TechniqueResult(
                success=False,
                payload={"error": "Retrieval failed or returned no hits"},
                meta={"method": "fid", "mode": self.mode}
            )

        hits = retrieval_result.payload["hits"]

        if not hits:
            return TechniqueResult(
                success=True,
                payload={
                    "answer": "No relevant documents found.",
                    "component_outputs": [],
                    "hits": []
                },
                meta={"method": "fid", "mode": self.mode}
            )

        # Step 2: Extract contexts from hits
        contexts = []
        for hit in hits:
            if hasattr(hit, 'chunk') and hasattr(hit.chunk, 'text'):
                context = hit.chunk.text
            elif hasattr(hit, 'text'):
                context = hit.text
            elif hasattr(hit, 'meta') and 'text' in hit.meta:
                context = hit.meta['text']
            else:
                context = str(hit)  # Fallback to string representation
            contexts.append(context)

        # Step 3: Generate responses based on mode
        component_outputs = []

        if self.mode == "separate":
            # Process each context separately
            for i, context in enumerate(contexts):
                prompt = f"Query: {query}\n\nContext: {context}\n\nAnswer:"
                output = self._generate_single(prompt)
                component_outputs.append({
                    "context_index": i,
                    "prompt": prompt,
                    "output": output
                })

            # Aggregate outputs
            aggregated_parts = [comp["output"] for comp in component_outputs]
            final_answer = "\n\n---\n\n".join(aggregated_parts)

        elif self.mode == "concat":
            # Concatenate all contexts
            all_contexts = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
            prompt = f"Query: {query}\n\n{all_contexts}\n\nAnswer:"
            output = self._generate_single(prompt)

            component_outputs.append({
                "contexts_count": len(contexts),
                "prompt": prompt,
                "output": output
            })
            final_answer = output

        else:
            return TechniqueResult(
                success=False,
                payload={"error": f"Unknown mode: {self.mode}"},
                meta={"method": "fid", "mode": self.mode}
            )

        return TechniqueResult(
            success=True,
            payload={
                "answer": final_answer,
                "component_outputs": component_outputs,
                "hits": hits
            },
            meta={"method": "fid", "mode": self.mode, "contexts_processed": len(contexts)}
        )

    def _generate_single(self, prompt: str) -> str:
        """Generate text for a single prompt using available generator or adapter."""
        if self.generator:
            result = self.generator.apply(prompt=prompt, **self.generator_kwargs)
            if result.success and "text" in result.payload:
                return result.payload["text"]
            else:
                return f"GENERATION_ERROR: {result.payload}"

        elif self.llm_adapter:
            try:
                return self.llm_adapter.generate(prompt, **self.generator_kwargs)
            except Exception as e:
                return f"ADAPTER_ERROR: {str(e)}"

        else:
            # Fallback behavior (shouldn't reach here due to constructor logic)
            return f"FALLBACK: {prompt}"
