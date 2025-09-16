"""Benchmark harness for evaluating RAG techniques."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from ..schemas import Document
from .loaders import QAItem, load_qa_dataset


class BenchmarkHarness:
    """
    Reusable benchmarking harness for evaluating RAG techniques.

    Supports running retrieval+generation pipelines on QA datasets
    and computing simple overlap-based metrics.
    """

    def __init__(self, quick_mode: bool = False, verbose: bool = False):
        """
        Initialize benchmark harness.

        Args:
            quick_mode: If True, limit to first few examples for fast testing
            verbose: If True, print progress information
        """
        self.quick_mode = quick_mode
        self.verbose = verbose

    def run_benchmark(
        self,
        technique,
        dataset_path: Union[str, Path],
        corpus_docs: list[Document],
        output_path: Union[str, Path] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run benchmark on a QA dataset using the given technique."""
        # Load dataset
        qa_items = load_qa_dataset(dataset_path)

        # Quick mode: limit examples
        if self.quick_mode:
            qa_items = qa_items[:3]
            if self.verbose:
                print(f"Quick mode: evaluating on {len(qa_items)} examples")

        if self.verbose:
            print(f"Running benchmark with {len(qa_items)} QA pairs")
            print(f"Technique: {technique.__class__.__name__}")
            print(f"Corpus size: {len(corpus_docs)}")

        # Run evaluation
        start_time = time.time()
        results = []

        for i, qa_item in enumerate(qa_items):
            if self.verbose and i % 10 == 0:
                print(f"Processing item {i + 1}/{len(qa_items)}")

            # Apply technique
            try:
                result = technique.apply(
                    query=qa_item.question,
                    corpus=corpus_docs,
                    top_k=top_k
                )

                # Handle different result types
                retrieved_docs = []
                if hasattr(result, 'retrieved_documents'):
                    retrieved_docs = result.retrieved_documents
                elif hasattr(result, 'documents'):
                    retrieved_docs = result.documents
                elif hasattr(result, 'payload') and result.payload:
                    if isinstance(result.payload, list):
                        retrieved_docs = result.payload
                    else:
                        retrieved_docs = [result.payload] if result.payload else []

                # Compute metrics
                metrics = self._compute_metrics(qa_item, retrieved_docs)

                results.append({
                    "question": qa_item.question,
                    "expected_answer": qa_item.answer,
                    "retrieved_count": len(retrieved_docs),
                    "metrics": metrics,
                    "metadata": qa_item.metadata
                })

            except Exception as e:
                if self.verbose:
                    print(f"Error processing item {i}: {e}")
                results.append({
                    "question": qa_item.question,
                    "expected_answer": qa_item.answer,
                    "error": str(e),
                    "retrieved_count": 0,
                    "metrics": {"exact_match": 0.0, "f1": 0.0, "overlap": 0.0},
                    "metadata": qa_item.metadata
                })

        end_time = time.time()

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(results)

        # Extract technique name (prefer meta.name if available)
        technique_name = technique.__class__.__name__
        if hasattr(technique, 'meta') and hasattr(technique.meta, 'name'):
            technique_name = technique.meta.name

        # Build final result
        benchmark_result = {
            "timestamp": datetime.now().isoformat(),
            "technique": technique_name,
            "dataset_path": str(dataset_path),
            "corpus_size": len(corpus_docs),
            "num_questions": len(qa_items),
            "quick_mode": self.quick_mode,
            "top_k": top_k,
            "runtime_seconds": end_time - start_time,
            "metrics": aggregated_metrics,
            "individual_results": results,
        }

        # Save results if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_result, f, indent=2)

            if self.verbose:
                print(f"Results saved to: {output_path}")

        if self.verbose:
            print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
            print(f"Average metrics: {aggregated_metrics}")

        return benchmark_result

    def _compute_metrics(
        self, qa_item: QAItem, retrieved_docs: list
    ) -> dict[str, float]:
        """Compute simple overlap-based metrics."""
        # Extract text from retrieved documents
        retrieved_text = ""
        for doc in retrieved_docs:
            if hasattr(doc, 'text'):
                retrieved_text += doc.text + " "
            elif isinstance(doc, str):
                retrieved_text += doc + " "
            else:
                retrieved_text += str(doc) + " "

        retrieved_text = retrieved_text.strip()
        expected_answer = qa_item.answer.strip()

        # Tokenize for comparison (simple word splitting)
        expected_tokens = set(expected_answer.lower().split())
        retrieved_tokens = set(retrieved_text.lower().split())

        # Exact match: check if expected answer appears in retrieved text
        exact_match = 1.0 if expected_answer.lower() in retrieved_text.lower() else 0.0

        # F1 score: token-level precision and recall
        intersection = expected_tokens.intersection(retrieved_tokens)
        if not expected_tokens:
            f1 = 1.0 if not retrieved_tokens else 0.0
        else:
            if not intersection:
                f1 = 0.0
            else:
                precision = (
                    len(intersection) / len(retrieved_tokens)
                    if retrieved_tokens else 0.0
                )
                recall = len(intersection) / len(expected_tokens)
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0.0
                )

        # Simple word overlap ratio
        if not expected_tokens:
            overlap = 0.0
        else:
            overlap = (
                len(expected_tokens.intersection(retrieved_tokens)) /
                len(expected_tokens)
            )

        return {
            "exact_match": exact_match,
            "f1": f1,
            "overlap": overlap,
        }

    def _aggregate_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate metrics across all results."""
        valid_results = [
            r for r in results
            if "metrics" in r and isinstance(r["metrics"], dict)
        ]

        if not valid_results:
            return {"exact_match": 0.0, "f1": 0.0, "overlap": 0.0}

        # Average each metric
        metric_names = ["exact_match", "f1", "overlap"]
        aggregated = {}

        for metric in metric_names:
            values = [r["metrics"].get(metric, 0.0) for r in valid_results]
            aggregated[metric] = sum(values) / len(values) if values else 0.0

        return aggregated
