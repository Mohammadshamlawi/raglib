#!/usr/bin/env python3
"""
Vector Retrieval Techniques Benchmark for RAGLib.

This script demonstrates and benchmarks all vector/dense retrieval techniques
available in RAGLib: FAISS retrieval, Dual Encoder, ColBERT, Multi-Query,
and Multi-Vector retrieval.

Usage:
    python vector_retrieval_benchmark.py              # Run all vector techniques
    python vector_retrieval_benchmark.py --quick      # Quick mode
    python vector_retrieval_benchmark.py --technique faiss_retriever  # Test single technique
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from raglib.adapters import DummyEmbedder
from raglib.registry import TechniqueRegistry
from raglib.schemas import Chunk, Document


def create_benchmark_corpus() -> list[Document]:
    """Create a comprehensive corpus for testing vector retrieval techniques."""
    return [
        Document(
            id="ai_overview",
            text="Artificial intelligence encompasses machine learning, deep learning, "
                 "natural language processing, computer vision, and robotics. Modern AI "
                 "systems use neural networks trained on massive datasets to perform "
                 "complex cognitive tasks previously requiring human intelligence."
        ),
        Document(
            id="ml_fundamentals", 
            text="Machine learning algorithms automatically learn patterns from data "
                 "without explicit programming. Supervised learning uses labeled data, "
                 "unsupervised learning finds hidden patterns, and reinforcement learning "
                 "optimizes through trial and error feedback mechanisms."
        ),
        Document(
            id="deep_learning",
            text="Deep learning utilizes artificial neural networks with multiple hidden "
                 "layers to model complex patterns in data. Convolutional neural networks "
                 "excel at image processing, while recurrent networks handle sequential "
                 "data like text and time series effectively."
        ),
        Document(
            id="nlp_advances",
            text="Natural language processing has revolutionized human-computer interaction "
                 "through transformer architectures like BERT and GPT. These models understand "
                 "context, generate coherent text, and perform tasks like translation, "
                 "summarization, and question answering with remarkable accuracy."
        ),
        Document(
            id="computer_vision",
            text="Computer vision enables machines to interpret and analyze visual information "
                 "from images and videos. Object detection, image classification, facial "
                 "recognition, and medical imaging applications demonstrate the power of "
                 "modern computer vision systems in real-world scenarios."
        ),
        Document(
            id="robotics_integration",
            text="Robotics combines artificial intelligence with mechanical engineering to "
                 "create autonomous systems. Service robots assist in healthcare, industrial "
                 "robots optimize manufacturing, and exploration robots venture into dangerous "
                 "environments like deep ocean and outer space missions."
        ),
        Document(
            id="ai_ethics",
            text="Artificial intelligence ethics addresses bias, fairness, transparency, "
                 "and accountability in AI systems. Responsible AI development requires "
                 "diverse teams, ethical guidelines, and continuous monitoring to prevent "
                 "harmful outcomes and ensure equitable benefits for society."
        ),
        Document(
            id="quantum_computing",
            text="Quantum computing leverages quantum mechanics principles like superposition "
                 "and entanglement to solve computationally complex problems. Quantum algorithms "
                 "promise exponential speedups for cryptography, optimization, and drug "
                 "discovery applications in the near future."
        ),
        Document(
            id="edge_computing",
            text="Edge computing brings computation closer to data sources, reducing latency "
                 "and bandwidth requirements. IoT devices, autonomous vehicles, and smart "
                 "cities benefit from real-time processing capabilities enabled by edge "
                 "infrastructure deployment strategies."
        ),
        Document(
            id="cybersecurity",
            text="Cybersecurity protects digital systems from malicious attacks using "
                 "advanced threat detection, encryption, and incident response protocols. "
                 "AI-powered security tools identify anomalies, predict threats, and "
                 "automate defensive measures against evolving cyber threats."
        )
    ]


def create_test_queries() -> list[dict[str, Any]]:
    """Create test queries with expected relevant documents."""
    return [
        {
            "query": "neural networks for image processing",
            "expected_relevant": ["deep_learning", "computer_vision", "ai_overview"],
            "description": "Technical query about specific AI techniques"
        },
        {
            "query": "machine learning patterns and data analysis",
            "expected_relevant": ["ml_fundamentals", "ai_overview", "deep_learning"],
            "description": "Broad query about ML fundamentals"
        },
        {
            "query": "human language understanding by computers",
            "expected_relevant": ["nlp_advances", "ai_overview"],
            "description": "Query about natural language processing"
        },
        {
            "query": "autonomous systems and robotics applications",
            "expected_relevant": ["robotics_integration", "edge_computing"],
            "description": "Query about robotics and automation"
        },
        {
            "query": "ethical considerations in AI development",
            "expected_relevant": ["ai_ethics"],
            "description": "Query about AI ethics and responsibility"
        },
        {
            "query": "quantum algorithms and computational speedup",
            "expected_relevant": ["quantum_computing"],
            "description": "Query about quantum computing applications"
        },
        {
            "query": "real-time processing and IoT systems",
            "expected_relevant": ["edge_computing", "robotics_integration"],
            "description": "Query about edge computing and IoT"
        },
        {
            "query": "threat detection and security automation",
            "expected_relevant": ["cybersecurity"],
            "description": "Query about AI in cybersecurity"
        }
    ]


def documents_to_chunks(documents: list[Document]) -> list[Chunk]:
    """Convert documents to chunks for retrieval."""
    return [
        Chunk(
            id=doc.id,
            text=doc.text,
            start_idx=0,
            end_idx=len(doc.text),
            doc_id=doc.id,
            meta=doc.meta
        )
        for doc in documents
    ]


def calculate_metrics(hits: list, expected_relevant: list[str], k: int = 5) -> dict[str, float]:
    """Calculate retrieval metrics."""
    if not hits:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0}
    
    # Get top-k results
    top_k_hits = hits[:k]
    retrieved_ids = [hit.doc_id for hit in top_k_hits]
    
    # Calculate precision and recall
    relevant_retrieved = set(retrieved_ids) & set(expected_relevant)
    precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(relevant_retrieved) / len(expected_relevant) if expected_relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate NDCG@k (simplified)
    dcg = 0.0
    for i, hit in enumerate(top_k_hits):
        if hit.doc_id in expected_relevant:
            dcg += 1.0 / (i + 2)  # +2 because log2(1) = 0
    
    # Ideal DCG
    idcg = sum(1.0 / (i + 2) for i in range(min(len(expected_relevant), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ndcg": ndcg
    }


def benchmark_technique(
    technique_name: str,
    documents: list[Document],
    test_queries: list[dict[str, Any]],
    embedder: DummyEmbedder,
    quick_mode: bool = False
) -> dict[str, Any]:
    """Benchmark a single vector retrieval technique."""
    print(f"\n{'='*20} {technique_name.upper()} {'='*20}")
    
    try:
        # Get technique class
        TechniqueClass = TechniqueRegistry.get(technique_name)
        if not TechniqueClass:
            return {"error": f"Technique {technique_name} not found"}
        
        # Initialize technique with appropriate parameters
        start_time = time.time()
        
        if technique_name == "faiss_retriever":
            technique = TechniqueClass(embedder=embedder, index_type="flat")
        elif technique_name == "dual_encoder":
            technique = TechniqueClass(
                query_embedder=embedder,
                doc_embedder=embedder,
                similarity="cosine"
            )
        elif technique_name == "colbert_retriever":
            technique = TechniqueClass(embedder=embedder, max_tokens=64)
        elif technique_name == "multi_query_retriever":
            # For demo, we'll use a simple base retriever
            base_retriever = TechniqueRegistry.get("faiss_retriever")(embedder=embedder)
            base_retriever.add_chunks(documents_to_chunks(documents))
            technique = TechniqueClass(
                base_retriever=base_retriever,
                num_queries=3,
                fusion_method="rrf"
            )
        elif technique_name == "multi_vector_retriever":
            technique = TechniqueClass(
                embedder=embedder,
                segment_size=100,
                aggregation_method="max"
            )
        else:
            technique = TechniqueClass(embedder=embedder)
        
        init_time = time.time() - start_time
        
        # Add documents (if technique supports it)
        if hasattr(technique, 'add_chunks') and technique_name != "multi_query_retriever":
            chunks = documents_to_chunks(documents)
            index_start = time.time()
            technique.add_chunks(chunks)
            index_time = time.time() - index_start
        else:
            index_time = 0.0
        
        # Run queries
        query_results = []
        total_query_time = 0.0
        
        test_set = test_queries[:3] if quick_mode else test_queries
        
        for query_data in test_set:
            query = query_data["query"]
            expected = query_data["expected_relevant"]
            
            query_start = time.time()
            
            if technique_name == "multi_query_retriever":
                # Multi-query needs special handling since it has its own retriever
                result = technique.apply(query=query, top_k=5)
            else:
                result = technique.apply(query=query, top_k=5)
            
            query_time = time.time() - query_start
            total_query_time += query_time
            
            if result.success:
                hits = result.payload.get("hits", [])
                metrics = calculate_metrics(hits, expected, k=5)
                
                query_results.append({
                    "query": query,
                    "query_time": query_time,
                    "num_results": len(hits),
                    "top_score": hits[0].score if hits else 0.0,
                    "metrics": metrics,
                    "top_results": [
                        {"doc_id": hit.doc_id, "score": hit.score}
                        for hit in hits[:3]
                    ]
                })
                
                print(f"Query: {query[:50]}...")
                print(f"  Results: {len(hits)}, Time: {query_time:.3f}s")
                print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
                print(f"  Top matches: {[hit.doc_id for hit in hits[:3]]}")
                
            else:
                print(f"Query failed: {result.error}")
                query_results.append({
                    "query": query,
                    "error": result.error,
                    "query_time": query_time
                })
        
        # Calculate aggregate metrics
        successful_queries = [r for r in query_results if "metrics" in r]
        if successful_queries:
            avg_metrics = {
                metric: sum(r["metrics"][metric] for r in successful_queries) / len(successful_queries)
                for metric in ["precision", "recall", "f1", "ndcg"]
            }
        else:
            avg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0}
        
        avg_query_time = total_query_time / len(test_set) if test_set else 0.0
        
        print(f"\nSummary for {technique_name}:")
        print(f"  Avg Precision: {avg_metrics['precision']:.3f}")
        print(f"  Avg Recall: {avg_metrics['recall']:.3f}")
        print(f"  Avg F1: {avg_metrics['f1']:.3f}")
        print(f"  Avg NDCG: {avg_metrics['ndcg']:.3f}")
        print(f"  Avg Query Time: {avg_query_time:.3f}s")
        
        return {
            "technique": technique_name,
            "success": True,
            "init_time": init_time,
            "index_time": index_time,
            "avg_query_time": avg_query_time,
            "total_queries": len(test_set),
            "successful_queries": len(successful_queries),
            "avg_metrics": avg_metrics,
            "query_results": query_results
        }
        
    except Exception as e:
        error_msg = f"Error testing {technique_name}: {str(e)}"
        print(error_msg)
        return {
            "technique": technique_name,
            "success": False,
            "error": error_msg
        }


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Vector Retrieval Benchmark")
    parser.add_argument(
        "--technique", 
        choices=["faiss_retriever", "dual_encoder", "colbert_retriever", 
                "multi_query_retriever", "multi_vector_retriever"],
        help="Run benchmark for specific technique only"
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode - fewer queries")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    print("Vector Retrieval Techniques Benchmark")
    print("=" * 50)
    
    # Prepare data
    documents = create_benchmark_corpus()
    test_queries = create_test_queries()
    embedder = DummyEmbedder(dim=384)
    
    print(f"Corpus: {len(documents)} documents")
    print(f"Test queries: {len(test_queries)}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    
    # Determine techniques to test
    if args.technique:
        techniques_to_test = [args.technique]
    else:
        # Get all vector retrieval techniques
        vector_techniques = []
        all_techniques = TechniqueRegistry.list()
        
        for name, tech_class in all_techniques.items():
            if hasattr(tech_class, 'meta'):
                category = getattr(tech_class.meta, 'category', '')
                if category in ['retrieval', 'retrieval_enhancement'] or name in [
                    'faiss_retriever', 'dual_encoder', 'colbert_retriever',
                    'multi_query_retriever', 'multi_vector_retriever'
                ]:
                    vector_techniques.append(name)
        
        techniques_to_test = vector_techniques
    
    print(f"Testing techniques: {techniques_to_test}")
    
    # Run benchmarks
    all_results = []
    start_time = time.time()
    
    for technique_name in techniques_to_test:
        result = benchmark_technique(
            technique_name, documents, test_queries, embedder, args.quick
        )
        all_results.append(result)
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*20} BENCHMARK SUMMARY {'='*20}")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Techniques tested: {len(techniques_to_test)}")
    
    successful_results = [r for r in all_results if r.get("success", False)]
    
    if successful_results:
        print("\nPerformance Comparison:")
        print(f"{'Technique':<20} {'Precision':<10} {'Recall':<8} {'F1':<8} {'NDCG':<8} {'Avg Time':<10}")
        print("-" * 70)
        
        for result in successful_results:
            metrics = result.get("avg_metrics", {})
            name = result["technique"]
            print(f"{name:<20} {metrics.get('precision', 0):<10.3f} {metrics.get('recall', 0):<8.3f} "
                  f"{metrics.get('f1', 0):<8.3f} {metrics.get('ndcg', 0):<8.3f} "
                  f"{result.get('avg_query_time', 0):<10.3f}s")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"vector_retrieval_benchmark_{timestamp}.json")
    
    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time": total_time,
        "corpus_size": len(documents),
        "num_queries": len(test_queries),
        "quick_mode": args.quick,
        "techniques_tested": techniques_to_test,
        "results": all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(benchmark_summary, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Exit with error code if any technique failed
    failed_techniques = [r["technique"] for r in all_results if not r.get("success", False)]
    if failed_techniques:
        print(f"\nFailed techniques: {failed_techniques}")
        sys.exit(1)
    else:
        print("\nAll techniques completed successfully!")


if __name__ == "__main__":
    main()