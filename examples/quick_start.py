#!/usr/bin/env python3
"""Quick start example for RAGLib.

This script demonstrates the basic usage of RAGLib with a simple
dense retrieval pipeline using default fallback adapters.
"""

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
    """Run the quick start example."""
    print("🚀 RAGLib Quick Start Example")
    print("=" * 40)
    
    # Sample documents
    documents = [
        ("RAGLib is a comprehensive library for building "
         "retrieval-augmented generation systems."),
        ("It provides a unified interface for different RAG "
         "techniques and components."),
        ("You can easily compose chunkers, retrievers, rerankers, "
         "and generators into pipelines."),
        ("The library supports both lightweight default adapters "
         "and heavy production adapters."),
        ("Each technique implements the same apply() method "
         "for consistency and composability.")
    ]
    
    print(f"📄 Working with {len(documents)} documents")
    
    # Step 1: Initialize chunker
    try:
        ChunkerClass = TechniqueRegistry.get("fixed_size_chunker")
        chunker = ChunkerClass(chunk_size=100, overlap=20)
        print("✅ Chunker initialized")
    except Exception as e:
        print(f"❌ Failed to initialize chunker: {e}")
        return 1
    
    # Step 2: Chunk documents
    try:
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks_result = chunker.apply(doc)
            if not chunks_result.success:
                print(f"❌ Chunking failed for document {i}: {chunks_result.error}")
                return 1
            all_chunks.extend(chunks_result.payload["chunks"])
        
        chunks = all_chunks
        print(f"📝 Created {len(chunks)} chunks")
        
        # Show first few chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"   Chunk {i+1}: {chunk.text[:60]}...")
    
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        return 1
    
    # Step 3: Initialize retriever with fallback adapters
    try:
        RetrieverClass = TechniqueRegistry.get("dense_retriever")
        embedder = DummyEmbedder(dim=384)
        vectorstore = InMemoryVectorStore()
        retriever = RetrieverClass(embedder=embedder, vectorstore=vectorstore)
        print("✅ Dense retriever initialized with fallback adapters")
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        return 1
    
    # Step 4: Index chunks
    try:
        index_result = retriever.apply(chunks)
        if not index_result.success:
            print(f"❌ Indexing failed: {index_result.error}")
            return 1

        indexed_count = index_result.payload.get("added", len(chunks))
        print(f"🔍 Indexed {indexed_count} chunks")

    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        return 1    # Step 5: Perform retrieval
    query = "How do you compose different techniques in RAGLib?"
    print(f"\n❓ Query: {query}")

    try:
        retrieve_result = retriever.apply(query=query, top_k=3)
        if not retrieve_result.success:
            print(f"❌ Retrieval failed: {retrieve_result.error}")
            return 1

        hits = retrieve_result.payload["hits"]
        print(f"📋 Retrieved {len(hits)} relevant chunks:")
        for i, hit in enumerate(hits):
            if hit.chunk:
                print(f"   {i+1}. (Score: {hit.score:.3f}) {hit.chunk.text}")
            else:
                print(f"   {i+1}. (Score: {hit.score:.3f}) [No chunk content]")

    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return 1
    
    # Step 6: Try generation (if available)
    try:
        GeneratorClass = TechniqueRegistry.get("llm_generator")
        generator = GeneratorClass()
        print("\n✅ Generator initialized")

        # Extract chunk texts for generation context
        context_chunks = [hit.chunk.text for hit in hits if hit.chunk]

        generate_result = generator.apply(query=query, context=context_chunks)
        if generate_result.success:
            answer = generate_result.payload.get("answer", "No answer generated")
            print(f"💡 Generated Answer: {answer}")
        else:
            print(f"⚠️  Generation failed (using fallback): {generate_result.error}")
            print("💡 Fallback Answer: Based on the retrieved context, you can "
                  "compose different techniques in RAGLib by using the unified "
                  "apply() method interface.")

    except Exception as e:
        print(f"⚠️  Generator not available: {e}")
        print("💡 Fallback Answer: Based on the retrieved context, you can "
              "compose different techniques in RAGLib by using the unified "
              "apply() method interface.")

    print("\n🎉 Quick start completed successfully!")
    print("\n📖 Next Steps:")
    print("   - Try 'raglib-cli run-example e2e_toy' for a complete pipeline")
    print("   - Explore the techniques catalog: raglib-cli docs-build")
    print("   - Check out examples/benchmark_run.py for evaluation")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
