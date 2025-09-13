# Getting Started

This guide will help you get up and running with RAGLib quickly.

## Installation

### Basic Installation

Install RAGLib using pip:

```bash
pip install raglib
```

This installs the core library with lightweight default adapters.

### Optional Dependencies

RAGLib supports optional dependencies for different use cases:

```bash
# For FAISS-based vector storage
pip install raglib[faiss]

# For LLM integrations (OpenAI, Transformers)
pip install raglib[llm]

# For development and testing
pip install raglib[dev]

# Install everything
pip install raglib[faiss,llm,dev]
```

## Quick Start

Let's build a simple RAG pipeline:

### 1. Basic Example

```python
from raglib.techniques import (
    FixedSizeChunker,
    DenseRetriever,
    LLMGenerator
)
from raglib.adapters import (
    InMemoryVectorStore,
    DummyEmbedder
)

# Initialize components
chunker = FixedSizeChunker(chunk_size=512, overlap=50)
embedder = DummyEmbedder(dimension=384)  # Fallback embedder
vectorstore = InMemoryVectorStore()
retriever = DenseRetriever(embedder=embedder, vectorstore=vectorstore)
generator = LLMGenerator()  # Uses fallback LLM

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

### 2. Using the CLI

RAGLib provides a command-line interface for quick experimentation:

```bash
# Run the quick start example
raglib-cli quick-start

# Run a specific example
raglib-cli run-example e2e_toy

# Build documentation
raglib-cli docs-build
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
