# RAGLib Documentation

Welcome to **RAGLib**, a comprehensive library of Retrieval-Augmented Generation (RAG) techniques with a unified `RAGTechnique.apply()` API for research and production environments.

![RAGLib Architecture](https://via.placeholder.com/200x100/2196F3/FFFFFF?text=RAGLib){ align=right width="200" }

## What is RAGLib? 🤔

RAGLib provides a modular, extensible framework for implementing and experimenting with different RAG techniques. Each technique follows a consistent interface, making it easy to:

- :material-compare: **Compare** different RAG approaches on your data
- :material-layers: **Compose** techniques into complex pipelines
- :material-puzzle: **Extend** the library with your own custom techniques
- :material-rocket: **Scale** from prototyping to production

## Key Features ✨

### :material-wrench: Unified Interface
All techniques implement the same `apply()` method, ensuring consistency across the library.

### :material-package: Modular Design
Mix and match components as needed. Lightweight core with optional heavy dependencies.

### :material-rocket: Production Ready
Built with scalability and performance in mind, suitable for both research and production.

### :material-test-tube: Extensible
Easy-to-use plugin system for adding new techniques and adapters.

### :material-chart-line: Benchmarking
Built-in tools for comparing techniques and measuring performance.

## Quick Start 🚀

```python
from raglib.techniques import DenseRetriever
from raglib.adapters import InMemoryVectorStore

# Create and apply a technique
technique = DenseRetriever(
    vectorstore=InMemoryVectorStore()
)

# Apply to your documents
results = technique.apply(
    documents=["Your document content..."],
    query="What is RAG?"
)
```

[Get Started →](getting_started.md){ .md-button .md-button--primary }

## Architecture Overview

RAGLib is organized into several key components:

### Core Components

- **`RAGTechnique`**: Base class for all techniques
- **`TechniqueRegistry`**: Central registry for technique discovery
- **`TechniqueMeta`**: Metadata for technique description and categorization

### Technique Categories

- **Chunking**: Split documents into processable segments
- **Retrieval**: Find relevant information from knowledge bases
- **Reranking**: Improve retrieval quality through reordering
- **Generation**: Produce final answers using retrieved context
- **Orchestration**: Coordinate multiple techniques in complex workflows

### Adapters

- **Embedders**: Convert text to vector representations
- **Vector Stores**: Store and retrieve embeddings efficiently
- **LLM Adapters**: Interface with different language models

## Quick Example

```python
from raglib.techniques import DenseRetriever, FixedSizeChunker
from raglib.adapters import InMemoryVectorStore, DummyEmbedder

# Initialize components
chunker = FixedSizeChunker(chunk_size=512)
embedder = DummyEmbedder()
vectorstore = InMemoryVectorStore()
retriever = DenseRetriever(embedder=embedder, vectorstore=vectorstore)

# Process documents
documents = ["Your documents here..."]
chunks = chunker.apply(documents)
retriever.apply(chunks.payload["chunks"], mode="index")

# Query
query = "What is the main topic?"
results = retriever.apply(query, mode="retrieve", top_k=5)
```

## Next Steps

- [Getting Started](getting_started.md): Set up RAGLib and run your first example
- [Techniques](techniques.md): Browse the complete catalog of available techniques
- [API Reference](api.md): Detailed API documentation

## Community & Contributing

RAGLib is an open-source project welcoming contributions from the community. Whether you're fixing bugs, adding new techniques, or improving documentation, we'd love to have you involved!

- 📖 Read our [Contributing Guide](https://github.com/your-org/raglib/blob/main/CONTRIBUTING.md)
- 🐛 Report issues on [GitHub](https://github.com/your-org/raglib/issues)
- 💬 Join discussions in [GitHub Discussions](https://github.com/your-org/raglib/discussions)
