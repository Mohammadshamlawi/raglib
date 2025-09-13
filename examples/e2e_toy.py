#!/usr/bin/env python3
"""End-to-end toy RAG example for RAGLib.

This script demonstrates a complete RAG pipeline with chunking,
indexing, retrieval, reranking, and generation using dummy adapters.
"""

import json
import sys
from pathlib import Path

# Add raglib to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from raglib.adapters import DummyEmbedder, InMemoryVectorStore
    from raglib.registry import TechniqueRegistry
except ImportError as e:
    print(f"Failed to import raglib: {e}")
    print("Make sure raglib is installed: pip install -e .")
    sys.exit(1)


def main():
    """Run the end-to-end toy example."""
    print("🎪 RAGLib End-to-End Toy RAG Pipeline")
    print("=" * 45)
    
    # Toy knowledge base
    knowledge_base = [
        "Machine learning is a subset of artificial intelligence that enables "
        "computers to learn and make decisions without being explicitly programmed.",
        
        "Deep learning uses neural networks with multiple layers to process "
        "complex patterns in data, mimicking how the human brain works.",
        
        "Natural language processing (NLP) helps computers understand, interpret, "
        "and generate human language in a valuable way.",
        
        "Computer vision enables machines to identify and analyze visual content "
        "such as images and videos using deep learning algorithms.",
        
        "Reinforcement learning is a type of machine learning where agents learn "
        "to make decisions by receiving rewards or penalties for their actions.",
        
        "Large language models like GPT are trained on vast amounts of text data "
        "to understand and generate human-like text responses.",
        
        "Retrieval-augmented generation (RAG) combines information retrieval "
        "with text generation to provide accurate and contextual answers."
    ]
    
    print(f"📚 Knowledge Base: {len(knowledge_base)} documents")
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is RAG and how is it useful?",
        "Tell me about computer vision"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} Query {i} {'='*20}")
        print(f"❓ Query: {query}")
        
        query_result = process_query(knowledge_base, query)
        results.append({
            "query": query,
            "result": query_result
        })
        
        if query_result["success"]:
            print("✅ Pipeline completed successfully")
            print(f"💡 Answer: {query_result['answer'][:100]}...")
        else:
            print(f"❌ Pipeline failed: {query_result['error']}")
    
    # Summary
    successful_queries = sum(1 for r in results if r["result"]["success"])
    print(f"\n{'='*50}")
    print(f"📊 Summary: {successful_queries}/{len(test_queries)} queries successful")
    
    # Save results
    output_file = Path(__file__).parent / "e2e_toy_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {output_file}")
    
    return 0 if successful_queries == len(test_queries) else 1


def process_query(documents, query):
    """Process a single query through the complete RAG pipeline."""
    try:
        # Step 1: Chunking
        ChunkerClass = TechniqueRegistry.get("fixed_size_chunker")
        chunker = ChunkerClass(chunk_size=200, overlap=50)
        
        chunks_result = chunker.apply(documents)
        if not chunks_result.success:
            return {
                "success": False,
                "error": f"Chunking failed: {chunks_result.error}"
            }
        
        chunks = chunks_result.payload["chunks"]
        print(f"  📝 Chunked into {len(chunks)} pieces")
        
        # Step 2: Dense retrieval
        embedder = DummyEmbedder(dimension=384)
        vectorstore = InMemoryVectorStore()
        
        RetrieverClass = TechniqueRegistry.get("dense_retriever")
        retriever = RetrieverClass(embedder=embedder, vectorstore=vectorstore)
        
        # Index chunks
        index_result = retriever.apply(chunks, mode="index")
        if not index_result.success:
            return {"success": False, "error": f"Indexing failed: {index_result.error}"}
        
        # Retrieve relevant chunks
        retrieve_result = retriever.apply(query, mode="retrieve", top_k=5)
        if not retrieve_result.success:
            return {"success": False, "error": f"Retrieval failed: {retrieve_result.error}"}
        
        relevant_chunks = retrieve_result.payload["chunks"]
        print(f"  🔍 Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Step 3: Reranking with MMR (if available)
        try:
            MMRClass = TechniqueRegistry.get("mmr")
            mmr = MMRClass(lambda_param=0.7, embedder=embedder)
            
            rerank_result = mmr.apply(
                query=query,
                candidates=relevant_chunks,
                top_k=3
            )
            
            if rerank_result.success:
                reranked_chunks = rerank_result.payload.get("reranked_chunks", relevant_chunks[:3])
                print(f"  🎯 Reranked to top {len(reranked_chunks)} chunks")
            else:
                reranked_chunks = relevant_chunks[:3]
                print("  ⚠️ Reranking failed, using top chunks")
                
        except Exception:
            reranked_chunks = relevant_chunks[:3]
            print("  ⚠️ MMR not available, using top chunks")
        
        # Step 4: Generation
        try:
            GeneratorClass = TechniqueRegistry.get("llm_generator")
            generator = GeneratorClass()
            
            generate_result = generator.apply(
                query=query,
                context=reranked_chunks
            )
            
            if generate_result.success:
                answer = generate_result.payload["answer"]
                print("  ✍️ Generated answer using LLM")
            else:
                # Fallback answer
                answer = create_fallback_answer(query, reranked_chunks)
                print("  ✍️ Generated fallback answer")
                
        except Exception:
            # Fallback answer
            answer = create_fallback_answer(query, reranked_chunks)
            print("  ✍️ Generated fallback answer (no LLM available)")
        
        return {
            "success": True,
            "chunks_count": len(chunks),
            "retrieved_count": len(relevant_chunks),
            "reranked_count": len(reranked_chunks),
            "answer": answer,
            "context": reranked_chunks
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_fallback_answer(query, context_chunks):
    """Create a simple fallback answer when LLM is not available."""
    # Simple keyword-based relevance
    query_lower = query.lower()
    
    # Find most relevant chunk
    best_chunk = ""
    max_overlap = 0
    
    for chunk in context_chunks:
        chunk_lower = chunk.lower()
        # Count word overlaps
        query_words = set(query_lower.split())
        chunk_words = set(chunk_lower.split())
        overlap = len(query_words & chunk_words)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_chunk = chunk
    
    if best_chunk:
        # Simple extractive answer
        sentences = best_chunk.split('.')
        if sentences:
            return sentences[0].strip() + '.'
    
    return "Based on the available information, I can provide relevant context but cannot generate a specific answer."


if __name__ == "__main__":
    sys.exit(main())
