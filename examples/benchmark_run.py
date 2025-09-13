#!/usr/bin/env python3
"""Benchmark script for RAGLib techniques.

This script runs a small benchmark over toy QA pairs and outputs
evaluation metrics to demonstrate RAGLib's evaluation capabilities.
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
    # Import techniques to ensure they are registered
    import raglib.techniques  # This triggers registration of all techniques
except ImportError as e:
    print(f"Failed to import raglib: {e}")
    print("Make sure raglib is installed: pip install -e .")
    sys.exit(1)


# Toy benchmark dataset
BENCHMARK_DATA = {
    "knowledge_base": [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning algorithms can learn patterns from data automatically.",
        "Deep neural networks have multiple layers of interconnected nodes.",
        "Natural language processing helps computers understand human language.",
        "RAGLib provides a unified interface for retrieval-augmented generation.",
        "Vector databases store high-dimensional embeddings efficiently.",
        "Transformer models use attention mechanisms for sequence processing.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "Cross-encoders rank document pairs for relevance scoring.",
        "Semantic chunking splits text based on meaning rather than length."
    ],
    
    "qa_pairs": [
        {
            "question": "What is Python known for?",
            "expected_answer": "simplicity",
            "expected_chunks": ["Python is a high-level programming language"]
        },
        {
            "question": "How do machine learning algorithms work?",
            "expected_answer": "learn patterns from data",
            "expected_chunks": ["Machine learning algorithms can learn patterns"]
        },
        {
            "question": "What is RAGLib?",
            "expected_answer": "unified interface for retrieval-augmented generation",
            "expected_chunks": ["RAGLib provides a unified interface"]
        },
        {
            "question": "How do transformers process sequences?",
            "expected_answer": "attention mechanisms",
            "expected_chunks": ["Transformer models use attention mechanisms"]
        }
    ]
}


def main():
    """Run the benchmark evaluation."""
    print("📊 RAGLib Benchmark Evaluation")
    print("=" * 40)
    
    # Configuration
    techniques_config = {
        "chunker": {
            "name": "fixed_size_chunker",
            "params": {"chunk_size": 100, "overlap": 20}
        },
        "retriever": {
            "name": "dense_retriever",
            "params": {}
        }
    }
    
    print(f"📝 Knowledge Base: {len(BENCHMARK_DATA['knowledge_base'])} documents")
    print(f"❓ Test Questions: {len(BENCHMARK_DATA['qa_pairs'])} QA pairs")
    print(f"⚙️  Configuration: {techniques_config}")
    
    # Run benchmark
    results = run_benchmark(
        BENCHMARK_DATA['knowledge_base'],
        BENCHMARK_DATA['qa_pairs'],
        techniques_config
    )
    
    # Check if benchmark failed completely
    if not results:
        print("❌ Benchmark failed - no results to process")
        return 1
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Display results
    display_results(results, metrics)
    
    # Save results
    output_file = Path(__file__).parent / "benchmark_results.json"
    save_results(results, metrics, output_file)
    
    return 0


def run_benchmark(knowledge_base, qa_pairs, config):
    """Run benchmark evaluation on QA pairs."""
    results = []
    
    # Initialize pipeline components
    try:
        # Chunker
        ChunkerClass = TechniqueRegistry.get(config["chunker"]["name"])
        chunker = ChunkerClass(**config["chunker"]["params"])
        
        # Retriever
        RetrieverClass = TechniqueRegistry.get(config["retriever"]["name"])
        embedder = DummyEmbedder(dim=384)
        vectorstore = InMemoryVectorStore()
        retriever = RetrieverClass(
            embedder=embedder, 
            vectorstore=vectorstore,
            **config["retriever"]["params"]
        )
        
        print("✅ Pipeline components initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return []
    
    # Index knowledge base
    try:
        chunks_result = chunker.apply(knowledge_base)
        if not chunks_result.success:
            print(f"❌ Chunking failed: {chunks_result.error}")
            return []
        
        chunks = chunks_result.payload["chunks"]
        
        index_result = retriever.apply(chunks)
        if not index_result.success:
            print(f"❌ Indexing failed: {index_result.error}")
            return []
        
        print(f"🔍 Indexed {len(chunks)} chunks")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        return []
    
    # Evaluate each QA pair
    for i, qa_pair in enumerate(qa_pairs, 1):
        print(f"\n--- Evaluating Question {i} ---")
        question = qa_pair["question"]
        expected_answer = qa_pair["expected_answer"]
        expected_chunks = qa_pair["expected_chunks"]
        
        print(f"Q: {question}")
        
        start_time = time.time()
        
        try:
            # Retrieve relevant chunks
            retrieve_result = retriever.apply(query=question, top_k=3)
            
            if not retrieve_result.success:
                print(f"❌ Retrieval failed: {retrieve_result.error}")
                results.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "expected_chunks": expected_chunks,
                    "retrieved_chunks": [],
                    "answer": "",
                    "success": False,
                    "error": retrieve_result.error,
                    "latency_ms": 0
                })
                continue
            
            hits = retrieve_result.payload["hits"]
            retrieved_chunks = [hit.chunk for hit in hits if hit.chunk]
            scores = [hit.score for hit in hits]
            
            # Simple answer generation (extractive)
            answer = generate_simple_answer(question, retrieved_chunks)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            print(f"A: {answer}")
            print(f"Retrieved: {len(retrieved_chunks)} chunks")
            print(f"Latency: {latency_ms}ms")
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "expected_chunks": expected_chunks,
                "retrieved_chunks": [chunk.text for chunk in retrieved_chunks],
                "retrieval_scores": scores,
                "answer": answer,
                "success": True,
                "error": None,
                "latency_ms": latency_ms
            })
            
        except Exception as e:
            print(f"❌ Question {i} failed: {e}")
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "expected_chunks": expected_chunks,
                "retrieved_chunks": [],
                "answer": "",
                "success": False,
                "error": str(e),
                "latency_ms": 0
            })
    
    return results


def generate_simple_answer(question, chunks):
    """Generate a simple extractive answer from retrieved chunks."""
    if not chunks:
        return "No relevant information found."
    
    # Simple keyword matching
    question_words = set(question.lower().split())
    
    best_chunk = ""
    max_overlap = 0
    
    for chunk in chunks:
        chunk_words = set(chunk.text.lower().split())
        overlap = len(question_words & chunk_words)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_chunk = chunk
    
    if best_chunk:
        # Extract first sentence as answer
        sentences = best_chunk.text.split('.')
        if sentences:
            return sentences[0].strip()
    
    return chunks[0].text  # Fallback to first chunk text


def calculate_metrics(results):
    """Calculate evaluation metrics."""
    total_questions = len(results)
    successful_questions = sum(1 for r in results if r["success"])
    
    # Retrieval metrics
    retrieval_precision_scores = []
    answer_relevance_scores = []
    latencies = []
    
    for result in results:
        if not result["success"]:
            continue
            
        # Simple precision: check if any expected chunk keywords appear
        retrieved = result["retrieved_chunks"]
        expected = result["expected_chunks"]
        
        if retrieved and expected:
            # Check keyword overlap
            expected_words = set()
            for exp_chunk in expected:
                # Handle both string and Chunk object formats
                if hasattr(exp_chunk, 'text'):
                    expected_words.update(exp_chunk.text.lower().split())
                else:
                    expected_words.update(exp_chunk.lower().split())
            
            retrieved_words = set()
            for ret_chunk in retrieved:
                # Handle both string and Chunk object formats
                if hasattr(ret_chunk, 'text'):
                    retrieved_words.update(ret_chunk.text.lower().split())
                else:
                    retrieved_words.update(ret_chunk.lower().split())
            
            if expected_words:
                precision = len(expected_words & retrieved_words) / len(expected_words)
                retrieval_precision_scores.append(precision)
        
        # Answer relevance (keyword overlap with expected answer)
        answer = result["answer"].lower()
        expected_answer = result["expected_answer"].lower()
        
        answer_words = set(answer.split())
        expected_words = set(expected_answer.split())
        
        if expected_words:
            relevance = len(answer_words & expected_words) / len(expected_words)
            answer_relevance_scores.append(relevance)
        
        latencies.append(result["latency_ms"])
    
    # Calculate averages
    avg_retrieval_precision = (
        sum(retrieval_precision_scores) / len(retrieval_precision_scores)
        if retrieval_precision_scores else 0.0
    )
    
    avg_answer_relevance = (
        sum(answer_relevance_scores) / len(answer_relevance_scores)
        if answer_relevance_scores else 0.0
    )
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    return {
        "total_questions": total_questions,
        "successful_questions": successful_questions,
        "success_rate": successful_questions / total_questions if total_questions > 0 else 0.0,
        "avg_retrieval_precision": avg_retrieval_precision,
        "avg_answer_relevance": avg_answer_relevance,
        "avg_latency_ms": avg_latency,
        "total_latency_ms": sum(latencies)
    }


def display_results(results, metrics):
    """Display benchmark results."""
    print(f"\n{'='*50}")
    print("📊 BENCHMARK RESULTS")
    print(f"{'='*50}")
    
    print(f"✅ Success Rate: {metrics['success_rate']:.1%} "
          f"({metrics['successful_questions']}/{metrics['total_questions']})")
    print(f"🔍 Avg Retrieval Precision: {metrics['avg_retrieval_precision']:.3f}")
    print(f"💡 Avg Answer Relevance: {metrics['avg_answer_relevance']:.3f}")
    print(f"⏱️ Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
    print(f"⏱️ Total Time: {metrics['total_latency_ms']:.0f}ms")
    
    print(f"\n📋 DETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        print(f"{status} Q{i}: {result['question'][:50]}...")
        if result["success"]:
            print(f"    Answer: {result['answer'][:60]}...")
            print(f"    Retrieved: {len(result['retrieved_chunks'])} chunks")


def save_results(results, metrics, output_file):
    """Save results to JSON file."""
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "detailed_results": results,
        "benchmark_config": {
            "knowledge_base_size": len(BENCHMARK_DATA['knowledge_base']),
            "qa_pairs_count": len(BENCHMARK_DATA['qa_pairs'])
        }
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")


if __name__ == "__main__":
    sys.exit(main())
