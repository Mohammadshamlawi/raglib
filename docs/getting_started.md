# Getting Started

This guide will help you get up and running with RAGLib quickly.

## Installation

### Basic Installation

Install RAGLib using pip:

```bash
pip install rag-techlib
```

This installs the core library with lightweight default adapters.

### Optional Dependencies

RAGLib supports optional dependencies for different use cases:

```bash
# For FAISS-based vector storage
pip install rag-techlib[faiss]

# For LLM integrations (OpenAI, Transformers)
pip install rag-techlib[llm]

# For development and testing
pip install rag-techlib[dev]

# Install everything
pip install rag-techlib[faiss,llm,dev]
```

## Quick Start

Let's build a simple RAG pipeline:

### 1. Basic Example

```python
from raglib.techniques import (
    FixedSizeChunker,
    DenseRetriever,
    HyDE
)
from raglib.adapters import (
    InMemoryVectorStore,
    DummyEmbedder
)

# Initialize components
chunker = FixedSizeChunker(chunk_size=512, overlap=50)
embedder = DummyEmbedder(dim=384)  # Fallback embedder
vectorstore = InMemoryVectorStore()
retriever = DenseRetriever(embedder=embedder, vectorstore=vectorstore)
# Note: HyDE requires an LLM adapter for query expansion

# Sample documents
documents = [
    "RAGLib is a library for building retrieval-augmented generation systems.",
    "It provides a unified interface for different RAG techniques.",
    "You can easily compose techniques into complex pipelines.",
]

# Step 1: Chunk documents
chunks_result = chunker.apply(documents)
chunks = chunks_result.payload["chunks"]

print(f"Created {len(chunks)} chunks")

# Step 2: Index chunks
index_result = retriever.apply(chunks, mode="index")
print(f"Indexed {index_result.payload['indexed_count']} chunks")

# Step 3: Retrieve relevant chunks
query = "What is RAGLib used for?"
retrieve_result = retriever.apply(query, mode="retrieve", top_k=3)
relevant_chunks = retrieve_result.payload["chunks"]

print(f"Retrieved {len(relevant_chunks)} relevant chunks")

# Step 4: Generate answer
generate_result = generator.apply(
    query=query,
    context=relevant_chunks
)

print(f"Generated answer: {generate_result.payload['answer']}")
```

### 2. Advanced Chunking Techniques

RAGLib includes several sophisticated chunking techniques optimized for different document types:

```python
from raglib.techniques import (
    ContentAwareChunker,
    RecursiveChunker,
    DocumentSpecificChunker,
    PropositionalChunker,
    ParentDocumentChunker
)
from raglib.schemas import Document

# Academic paper with hierarchical structure
document = Document(
    id="research_paper",
    text="""
# Abstract

This paper presents a novel approach to information retrieval.

## Introduction

Information retrieval has evolved significantly with the advent of neural networks.

### Background

Previous work has focused on traditional keyword-based approaches.

## Methodology

We propose a hybrid approach combining neural and symbolic methods.
    """,
    meta={"type": "academic", "domain": "computer_science"}
)

# Content-aware chunking respects document structure
content_chunker = ContentAwareChunker(chunk_size=300, overlap=50)
result = content_chunker.apply(document)
content_chunks = result.payload["chunks"]

print(f"Content-aware chunking: {len(content_chunks)} chunks")
for chunk in content_chunks[:2]:
    print(f"  - Chunk: {chunk.text[:100]}...")

# Recursive chunking with hierarchical splitting
recursive_chunker = RecursiveChunker(
    chunk_size=250,
    overlap=30,
    separators=["\n\n", "\n", ". ", " "]  # Custom separator hierarchy
)
result = recursive_chunker.apply(document)
recursive_chunks = result.payload["chunks"]

print(f"Recursive chunking: {len(recursive_chunks)} chunks")

# Document-specific chunking adapts to document type
doc_chunker = DocumentSpecificChunker(chunk_size=400, overlap=40)
result = doc_chunker.apply(document)
doc_chunks = result.payload["chunks"]

print(f"Document-specific chunking: {len(doc_chunks)} chunks")

# Propositional chunking focuses on semantic units
prop_chunker = PropositionalChunker(chunk_size=200, overlap=20)
result = prop_chunker.apply(document)
prop_chunks = result.payload["chunks"]

print(f"Propositional chunking: {len(prop_chunks)} chunks")

# Parent-document chunking maintains context hierarchy
parent_chunker = ParentDocumentChunker(
    child_chunk_size=150,
    parent_chunk_size=600,
    overlap=25
)
result = parent_chunker.apply(document)
parent_data = result.payload

print(f"Parent-document chunking:")
print(f"  - Child chunks: {len(parent_data['child_chunks'])}")
print(f"  - Parent chunks: {len(parent_data['parent_chunks'])}")
```

### 3. Comparing Chunking Strategies

```python
from raglib.registry import TechniqueRegistry

# Get all chunking techniques
chunking_techniques = TechniqueRegistry.list_by_category("chunking")

# Test document
test_doc = Document(
    id="test",
    text="This is a test document. It has multiple sentences and paragraphs.\n\nThis is the second paragraph with more content for testing purposes.",
    meta={"source": "test"}
)

# Compare different chunking approaches
print("Chunking Strategy Comparison:")
print("-" * 40)

for name, technique_class in chunking_techniques.items():
    try:
        chunker = technique_class(chunk_size=100, overlap=20)
        result = chunker.apply(test_doc)
        
        if result.success:
            chunks = result.payload["chunks"]
            avg_length = sum(len(c.text) for c in chunks) / len(chunks)
            
            print(f"{name}:")
            print(f"  - Chunks: {len(chunks)}")
            print(f"  - Avg length: {avg_length:.1f} chars")
            print(f"  - Coverage: {result.meta.get('coverage_ratio', 'N/A')}")
        else:
            print(f"{name}: Failed - {result.error}")
            
    except Exception as e:
        print(f"{name}: Error - {e}")
    
    print()
```

### 4. Using the CLI

RAGLib provides a command-line interface for quick experimentation:

```bash
# Run the quick start example
raglib-cli quick-start

# Run a specific example
raglib-cli run-example e2e_toy

# Test all chunking techniques
python examples/chunking_benchmark.py

# Build documentation
raglib-cli docs-build

# List all available techniques
python -c "from raglib.registry import TechniqueRegistry; print('\\n'.join(TechniqueRegistry.list().keys()))"
```

## Core Concepts

### RAGTechnique Interface

All techniques in RAGLib implement the same interface:

```python
from raglib.core import RAGTechnique

class MyTechnique(RAGTechnique):
    def apply(self, *args, **kwargs):
        # Your implementation here
        return TechniqueResult(
            success=True,
            payload={"result": "your_data"}
        )
```

### TechniqueResult

Every technique returns a `TechniqueResult` object:

```python
result = technique.apply(data)

if result.success:
    data = result.payload
    print(f"Operation succeeded: {data}")
else:
    print(f"Operation failed: {result.error}")
```

### Registration System

Techniques are automatically discoverable through the registry:

```python
from raglib.registry import TechniqueRegistry

# List all registered techniques
techniques = TechniqueRegistry.list()
print(techniques.keys())

# Get a specific technique
ChunkerClass = TechniqueRegistry.get("fixed_size_chunker")
chunker = ChunkerClass(chunk_size=256)
```

## Working with Adapters

Adapters provide interfaces to external services and libraries:

### Embedders

```python
from raglib.adapters import DummyEmbedder

# Fallback embedder (no external dependencies)
embedder = DummyEmbedder(dimension=384)

# With sentence-transformers (requires llm extras)
# from raglib.adapters import SentenceTransformerEmbedder
# embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
```

### Vector Stores

```python
from raglib.adapters import InMemoryVectorStore

# In-memory storage (good for development)
vectorstore = InMemoryVectorStore()

# With FAISS (requires faiss extras)
# from raglib.adapters import FaissVectorStore
# vectorstore = FaissVectorStore(dimension=384)
```

## Configuration and Environment

### Environment Variables

RAGLib respects standard environment variables:

```bash
# OpenAI API key (for LLM generators)
export OPENAI_API_KEY="your-api-key"

# Hugging Face token (for some models)
export HF_TOKEN="your-token"
```

### Configuration Files

You can use configuration files to manage complex setups:

```yaml
# raglib_config.yaml
chunking:
  technique: "fixed_size_chunker"
  chunk_size: 512
  overlap: 50

retrieval:
  technique: "dense_retriever"
  top_k: 5
  
generation:
  technique: "llm_generator"
  model: "gpt-3.5-turbo"
```

## Next Steps

Now that you have RAGLib running:

1. **Explore Techniques**: Check out the [techniques catalog](techniques.md)
2. **Build Pipelines**: Learn about composing techniques
3. **Add Custom Techniques**: Extend RAGLib with your own implementations
4. **Optimize Performance**: Learn about production deployment strategies

## Getting Help

- **Documentation**: Browse the complete [API reference](api.md)
- **Examples**: Check out the `examples/` directory in the repository
- **Issues**: Report bugs or request features on [GitHub](https://github.com/your-org/raglib/issues)
- **Discussions**: Join community discussions for help and ideas
