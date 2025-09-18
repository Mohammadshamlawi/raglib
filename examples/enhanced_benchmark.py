#!/usr/bin/env python3
"""Enhanced Benchmark script for RAGLib techniques including vector retrieval.

This script runs benchmarks over multiple retrieval techniques including
the new vector/dense retrieval methods to compare their performance.
"""

import json
import sys
import time
from pathlib import Path

# Add raglib to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from raglib.adapters import DummyEmbedder, InMemoryVectorStore
    from raglib.registry import TechniqueRegistry
    from raglib.schemas import Document, Chunk
    # Import techniques to ensure they are registered
    import raglib.techniques  # This triggers registration of all techniques
except ImportError as e:
    print(f"Failed to import raglib: {e}")
    print("Make sure raglib is installed: pip install -e .")
    sys.exit(1)


# Enhanced benchmark dataset with more diverse content
BENCHMARK_DATA = {
    "knowledge_base": [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning algorithms can automatically learn patterns from large datasets.",
        "Deep neural networks have multiple layers of interconnected computational nodes.",
        "Natural language processing enables computers to understand and generate human language.",
        "RAGLib provides a unified interface for building retrieval-augmented generation systems.",
        "Vector databases efficiently store and search high-dimensional embedding representations.",
        "Transformer models use self-attention mechanisms for processing sequential data.",
        "Fine-tuning adapts pre-trained language models to domain-specific tasks.",
        "Cross-encoder models score document-query pairs for relevance ranking.",
        "Semantic chunking divides text based on conceptual meaning rather than fixed lengths.",
        "FAISS is a library for efficient similarity search over dense vector collections.",
        "ColBERT performs late interaction between query and document token representations.",
        "Multi-vector retrieval segments documents for improved granular matching.",
        "Dual encoders use separate models for encoding queries and documents asymmetrically.",
        "Reciprocal rank fusion combines multiple ranked lists into a single ranking."
    ],
    
    "qa_pairs": [
        {
            "question": "What is Python known for?",
            "expected_answer": "simplicity",
            "expected_chunks": ["Python is a high-level programming language"]
        },
        {
            "question": "How do machine learning algorithms work?",
            "expected_answer": "learn patterns",
            "expected_chunks": ["Machine learning algorithms can automatically learn"]
        },
        {
            "question": "What are neural networks?",
            "expected_answer": "layers",
            "expected_chunks": ["Deep neural networks have multiple layers"]
        },
        {
            "question": "What does NLP do?",
            "expected_answer": "understand language",
            "expected_chunks": ["Natural language processing enables computers"]
        },
        {
            "question": "What is RAGLib?",
            "expected_answer": "unified interface",
            "expected_chunks": ["RAGLib provides a unified interface"]
        },
        {
            "question": "How do vector databases work?",
            "expected_answer": "store embeddings",
            "expected_chunks": ["Vector databases efficiently store"]
        },
        {
            "question": "What do transformers use?",
            "expected_answer": "attention",
            "expected_chunks": ["Transformer models use self-attention"]
        },
        {
            "question": "What is FAISS used for?",
            "expected_answer": "similarity search",
            "expected_chunks": ["FAISS is a library for efficient similarity search"]
        },
        {
            "question": "How does ColBERT work?",
            "expected_answer": "late interaction",
            "expected_chunks": ["ColBERT performs late interaction"]
        },
        {
            "question": "What is multi-vector retrieval?",
            "expected_answer": "segments documents",
            "expected_chunks": ["Multi-vector retrieval segments documents"]
        }
    ]
}


def get_technique_configs():
    """Get configurations for different retrieval techniques to benchmark."""
    return {
        "dense_retriever": {
            "chunker": {"name": "fixed_size_chunker", "params": {"chunk_size": 128, "overlap": 20}},
            "retriever": {"name": "dense_retriever", "params": {}}
        },
        "faiss_retriever": {
            "chunker": {"name": "fixed_size_chunker", "params": {"chunk_size": 128, "overlap": 20}},
            "retriever": {"name": "faiss_retriever", "params": {"index_type": "flat"}}
        },
        "dual_encoder": {
            "chunker": {"name": "fixed_size_chunker", "params": {"chunk_size": 128, "overlap": 20}},
            "retriever": {"name": "dual_encoder", "params": {"similarity": "cosine"}}
        },
        "colbert_retriever": {
            "chunker": {"name": "fixed_size_chunker", "params": {"chunk_size": 128, "overlap": 20}},
            "retriever": {"name": "colbert_retriever", "params": {"max_tokens": 64}}
        },
        "multi_vector_retriever": {
            "chunker": {"name": "fixed_size_chunker", "params": {"chunk_size": 128, "overlap": 20}},
            "retriever": {"name": "multi_vector_retriever", "params": {"segment_size": 50, "aggregation_method": "max"}}
        },
        "bm25": {
            "chunker": {"name": "fixed_size_chunker", "params": {"chunk_size": 128, "overlap": 20}},
            "retriever": {"name": "bm25", "params": {}}
        }
    }


def calculate_metrics(results):
    """Calculate evaluation metrics from benchmark results."""
    total_questions = len(results)
    if total_questions == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "success_rate": 0.0}
    
    total_precision = 0.0
    total_recall = 0.0
    successful_retrievals = 0
    
    for result in results:
        if result["retrieval_success"]:
            successful_retrievals += 1
            total_precision += result["precision"]
            total_recall += result["recall"]
    
    avg_precision = total_precision / total_questions if total_questions > 0 else 0.0
    avg_recall = total_recall / total_questions if total_questions > 0 else 0.0
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
    success_rate = successful_retrievals / total_questions if total_questions > 0 else 0.0
    
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": f1,
        "success_rate": success_rate,
        "total_questions": total_questions,
        "successful_retrievals": successful_retrievals
    }


def documents_to_chunks(documents):
    """Convert string documents to Chunk objects."""
    chunks = []
    for i, doc_text in enumerate(documents):
        chunk = Chunk(
            id=f"chunk_{i}",
            text=doc_text,
            start_idx=0,
            end_idx=len(doc_text),
            doc_id=f"doc_{i}"
        )
        chunks.append(chunk)
    return chunks


def run_technique_benchmark(technique_name, knowledge_base, qa_pairs, config):
    """Run benchmark for a specific technique."""
    print(f"\n--- Benchmarking {technique_name.upper()} ---")
    
    results = []
    start_time = time.time()
    
    try:
        # Initialize pipeline components
        ChunkerClass = TechniqueRegistry.get(config["chunker"]["name"])
        chunker = ChunkerClass(**config["chunker"]["params"])
        
        RetrieverClass = TechniqueRegistry.get(config["retriever"]["name"])
        
        # Initialize retriever based on type
        if technique_name in ["dense_retriever", "faiss_retriever"]:
            embedder = DummyEmbedder(dim=384)
            if technique_name == "dense_retriever":
                vectorstore = InMemoryVectorStore()
                retriever = RetrieverClass(embedder=embedder, vectorstore=vectorstore, **config["retriever"]["params"])
            else:  # faiss_retriever
                retriever = RetrieverClass(embedder=embedder, **config["retriever"]["params"])
        elif technique_name == "dual_encoder":
            embedder = DummyEmbedder(dim=384)
            retriever = RetrieverClass(
                query_embedder=embedder,
                doc_embedder=embedder,
                **config["retriever"]["params"]
            )
        elif technique_name in ["colbert_retriever", "multi_vector_retriever"]:
            embedder = DummyEmbedder(dim=384)
            retriever = RetrieverClass(embedder=embedder, **config["retriever"]["params"])
        elif technique_name == "bm25":
            # BM25 works with documents directly
            docs = [Document(id=f"doc_{i}", text=text) for i, text in enumerate(knowledge_base)]
            retriever = RetrieverClass(docs=docs, **config["retriever"]["params"])
        else:
            raise ValueError(f"Unknown technique: {technique_name}")
        
        print(f"✅ Initialized {technique_name}")
        
        # Index knowledge base (except for BM25 which indexes in constructor)
        if technique_name != "bm25":
            if technique_name == "dense_retriever":
                # Dense retriever needs documents converted to chunks first
                chunks = documents_to_chunks(knowledge_base)
                index_result = retriever.apply(chunks)
            else:
                # Other techniques use add_chunks method
                chunks = documents_to_chunks(knowledge_base)
                retriever.add_chunks(chunks)
                index_result = type('MockResult', (), {'success': True})()
            
            if not index_result.success:
                print(f"❌ Indexing failed for {technique_name}")
                return [], {"error": "indexing_failed"}
        
        print(f"✅ Indexed {len(knowledge_base)} documents")
        
        # Run queries
        for qa in qa_pairs:
            question = qa["question"]
            expected_chunks = qa["expected_chunks"]
            
            try:
                retrieve_result = retriever.apply(query=question, top_k=3)
                
                if retrieve_result.success:
                    hits = retrieve_result.payload.get("hits", [])
                    retrieved_texts = []
                    
                    # Extract text from hits
                    for hit in hits:
                        if technique_name == "bm25":
                            # BM25 returns document IDs
                            doc_idx = int(hit.doc_id.split("_")[-1])
                            retrieved_texts.append(knowledge_base[doc_idx])
                        else:
                            # Vector techniques return chunk IDs
                            chunk_idx = int(hit.doc_id.split("_")[-1])
                            retrieved_texts.append(knowledge_base[chunk_idx])
                    
                    # Calculate precision and recall
                    relevant_retrieved = 0
                    for expected in expected_chunks:
                        for retrieved in retrieved_texts:
                            if expected.lower() in retrieved.lower():
                                relevant_retrieved += 1
                                break
                    
                    precision = relevant_retrieved / len(retrieved_texts) if retrieved_texts else 0.0
                    recall = relevant_retrieved / len(expected_chunks) if expected_chunks else 0.0
                    
                    results.append({
                        "question": question,
                        "retrieval_success": True,
                        "num_results": len(hits),
                        "precision": precision,
                        "recall": recall,
                        "top_score": hits[0].score if hits else 0.0,
                        "retrieved_texts": retrieved_texts[:3]  # Top 3 for inspection
                    })
                    
                else:
                    results.append({
                        "question": question,
                        "retrieval_success": False,
                        "error": retrieve_result.error,
                        "precision": 0.0,
                        "recall": 0.0
                    })
                    
            except Exception as e:
                results.append({
                    "question": question,
                    "retrieval_success": False,
                    "error": str(e),
                    "precision": 0.0,
                    "recall": 0.0
                })
        
        execution_time = time.time() - start_time
        metrics = calculate_metrics(results)
        metrics["execution_time"] = execution_time
        
        print(f"✅ Completed {technique_name} benchmark")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1: {metrics['f1']:.3f}")
        print(f"   Success Rate: {metrics['success_rate']:.3f}")
        print(f"   Execution Time: {execution_time:.2f}s")
        
        return results, metrics
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Failed to benchmark {technique_name}: {e}")
        return [], {"error": str(e), "execution_time": execution_time}


def display_comparison_results(all_results):
    """Display comparison results across all techniques."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"{'Technique':<20} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Success':<8} {'Time':<8}")
    print("-" * 80)
    
    for technique_name, (results, metrics) in all_results.items():
        if "error" not in metrics:
            print(f"{technique_name:<20} {metrics['precision']:<10.3f} {metrics['recall']:<8.3f} "
                  f"{metrics['f1']:<8.3f} {metrics['success_rate']:<8.3f} {metrics['execution_time']:<8.2f}s")
        else:
            print(f"{technique_name:<20} {'ERROR':<10} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} "
                  f"{metrics['execution_time']:<8.2f}s")
    
    print("\n" + "=" * 80)


def main():
    """Main benchmark execution."""
    print("RAGLib Enhanced Retrieval Techniques Benchmark")
    print("=" * 60)
    
    # Get data and configurations
    knowledge_base = BENCHMARK_DATA["knowledge_base"]
    qa_pairs = BENCHMARK_DATA["qa_pairs"]
    technique_configs = get_technique_configs()
    
    print(f"📚 Knowledge Base: {len(knowledge_base)} documents")
    print(f"❓ Test Questions: {len(qa_pairs)} QA pairs")
    print(f"🔧 Techniques: {', '.join(technique_configs.keys())}")
    
    # Run benchmarks for all techniques
    all_results = {}
    total_start_time = time.time()
    
    for technique_name, config in technique_configs.items():
        try:
            # Check if technique is available
            if TechniqueRegistry.get(technique_name.split('_')[0] if '_' in technique_name else technique_name):
                results, metrics = run_technique_benchmark(technique_name, knowledge_base, qa_pairs, config)
                all_results[technique_name] = (results, metrics)
            else:
                print(f"⚠️  Technique {technique_name} not available, skipping...")
        except Exception as e:
            print(f"❌ Error with {technique_name}: {e}")
            all_results[technique_name] = ([], {"error": str(e), "execution_time": 0.0})
    
    total_execution_time = time.time() - total_start_time
    
    # Display comparison results
    display_comparison_results(all_results)
    
    # Save detailed results
    output_file = Path(__file__).parent / f"enhanced_benchmark_results_{int(time.time())}.json"
    
    # Prepare data for JSON serialization
    json_results = {}
    for technique_name, (results, metrics) in all_results.items():
        json_results[technique_name] = {
            "metrics": metrics,
            "results": results
        }
    
    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_execution_time": total_execution_time,
        "knowledge_base_size": len(knowledge_base),
        "num_questions": len(qa_pairs),
        "techniques": list(technique_configs.keys()),
        "results": json_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(benchmark_summary, f, indent=2)
    
    print(f"📊 Detailed results saved to: {output_file}")
    print(f"⏱️  Total execution time: {total_execution_time:.2f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())