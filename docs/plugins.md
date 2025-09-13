# RAGLib Plugin System

RAGLib supports a powerful plugin architecture that allows you to extend the framework with custom techniques, adapters, and other components. This guide explains how to create, distribute, and use plugins with RAGLib.

## Overview

The plugin system enables:

- **Technique plugins**: Custom RAG techniques and processing components
- **Adapter plugins**: New data sources and storage backends  
- **Architecture plugins**: Novel RAG architectures and patterns
- **Automatic discovery**: Plugins are automatically found and registered
- **Entry point support**: Standard Python packaging with setuptools
- **Local development**: Load plugins from local directories

## Quick Start

### Creating Your First Plugin

Create a new Python package with your custom technique:

```python
# my_plugin/techniques.py
from raglib.techniques.base import RAGTechnique
from raglib.schemas import Document, RagResult

class CustomRetriever(RAGTechnique):
    """Example custom retrieval technique."""
    
    def __init__(self, similarity_threshold=0.7):
        super().__init__()
        self.similarity_threshold = similarity_threshold
    
    def process(self, query: str, docs: list[Document]) -> RagResult:
        # Your custom retrieval logic
        filtered_docs = []
        for doc in docs:
            if self._compute_similarity(query, doc.text) > self.similarity_threshold:
                filtered_docs.append(doc)
        
        return RagResult(
            documents=filtered_docs,
            metadata={"threshold": self.similarity_threshold}
        )
    
    def _compute_similarity(self, query: str, text: str) -> float:
        # Simple word overlap example
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
            
        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)
```

### Plugin Package Structure

```
my_raglib_plugin/
├── setup.py                 # Package configuration with entry points
├── my_raglib_plugin/
│   ├── __init__.py          # Plugin initialization
│   ├── techniques.py        # Custom techniques
│   ├── adapters.py          # Custom adapters (optional)
│   └── utils.py            # Utility functions (optional)
└── tests/
    └── test_techniques.py   # Plugin tests
```

### setup.py Configuration

Configure entry points for automatic discovery:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="my-raglib-plugin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "raglib>=0.1.0",
    ],
    entry_points={
        # Technique plugins
        "raglib.techniques": [
            "custom_retriever = my_raglib_plugin.techniques:CustomRetriever",
            "custom_ranker = my_raglib_plugin.techniques:CustomRanker",
        ],
        # Adapter plugins (optional)
        "raglib.adapters": [
            "custom_vectorstore = my_raglib_plugin.adapters:CustomVectorStore",
        ],
        # Architecture plugins (optional)
        "raglib.architectures": [
            "custom_rag = my_raglib_plugin.architectures:CustomRAGArchitecture",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Custom RAGLib plugin with specialized techniques",
    python_requires=">=3.8",
)
```

### Installing and Using Plugins

```bash
# Install your plugin
pip install my-raglib-plugin

# Or install in development mode
pip install -e /path/to/my-raglib-plugin
```

```python
# Use your plugin techniques
from raglib.plugins import PluginLoader
from raglib.registry import TechniqueRegistry

# Discover and register plugins
loader = PluginLoader()
loader.discover()

# Your custom technique is now available
technique = TechniqueRegistry.get("custom_retriever")
result = technique.process("sample query", documents)
```

## Plugin Development Guide

### Technique Plugins

Custom techniques should inherit from the appropriate base class:

```python
from raglib.techniques.base import RAGTechnique, Chunker, Retriever, Generator

class MyCustomChunker(Chunker):
    """Custom document chunking strategy."""
    
    def process(self, text: str) -> list[Document]:
        # Your chunking logic
        chunks = self._smart_chunk(text)
        
        return [
            Document(id=f"chunk_{i}", text=chunk)
            for i, chunk in enumerate(chunks)
        ]
    
    def _smart_chunk(self, text: str) -> list[str]:
        # Custom chunking algorithm
        return text.split('\n\n')  # Simple example

class MyCustomGenerator(Generator):
    """Custom response generation."""
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        super().__init__()
        self.model_name = model_name
    
    def process(self, query: str, docs: list[Document]) -> RagResult:
        # Your generation logic
        context = "\n".join(doc.text for doc in docs)
        response = self._generate_response(query, context)
        
        return RagResult(
            response=response,
            documents=docs,
            metadata={"model": self.model_name}
        )
    
    def _generate_response(self, query: str, context: str) -> str:
        # Your LLM integration
        return f"Generated response for: {query}"
```

### Adapter Plugins

Create custom data adapters for different sources:

```python
from raglib.adapters.base import VectorStore, Embedder

class RedisVectorStore(VectorStore):
    """Redis-based vector storage."""
    
    def __init__(self, host="localhost", port=6379):
        import redis
        self.client = redis.Redis(host=host, port=port)
    
    def store_embeddings(self, docs: list[Document], embeddings: list[list[float]]):
        for doc, embedding in zip(docs, embeddings):
            self.client.hset(
                f"doc:{doc.id}",
                mapping={
                    "text": doc.text,
                    "embedding": json.dumps(embedding),
                    "metadata": json.dumps(doc.metadata or {})
                }
            )
    
    def similarity_search(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        # Your similarity search implementation
        # This would typically use Redis search capabilities
        pass

class HuggingFaceEmbedder(Embedder):
    """HuggingFace transformer embeddings."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query])[0].tolist()
```

### Architecture Plugins

Implement complete RAG architectures:

```python
from raglib.architectures.base import RAGArchitecture

class MultiStepRAG(RAGArchitecture):
    """Multi-step reasoning RAG architecture."""
    
    def __init__(self, retriever, reranker, generator):
        self.retriever = retriever
        self.reranker = reranker  
        self.generator = generator
    
    def process(self, query: str, docs: list[Document]) -> RagResult:
        # Step 1: Initial retrieval
        retrieval_result = self.retriever.process(query, docs)
        
        # Step 2: Re-ranking
        rerank_result = self.reranker.process(query, retrieval_result.documents)
        
        # Step 3: Multi-step generation
        steps = self._decompose_query(query)
        final_result = None
        
        for step in steps:
            step_result = self.generator.process(step, rerank_result.documents)
            # Accumulate results...
            final_result = step_result
        
        return final_result
    
    def _decompose_query(self, query: str) -> list[str]:
        # Query decomposition logic
        return [query]  # Simplified
```

## Plugin Distribution

### Publishing to PyPI

```bash
# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

### Version Compatibility

Specify RAGLib version requirements:

```python
# setup.py
install_requires=[
    "raglib>=0.1.0,<0.2.0",  # Compatible with 0.1.x
]
```

### Plugin Metadata

Add rich metadata for discovery:

```python
setup(
    name="my-raglib-plugin",
    # ... other setup args ...
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="rag, retrieval, augmented, generation, nlp",
    project_urls={
        "Homepage": "https://github.com/yourusername/my-raglib-plugin",
        "Documentation": "https://my-raglib-plugin.readthedocs.io/",
        "Bug Reports": "https://github.com/yourusername/my-raglib-plugin/issues",
    },
)
```

## Local Development

### Development Installation

```python
# Install RAGLib in development mode
pip install -e /path/to/raglib

# Install your plugin in development mode
pip install -e /path/to/my-plugin
```

### Local Plugin Discovery

Load plugins from local directories without installation:

```python
from raglib.plugins import PluginLoader

# Load plugins from specific directories
loader = PluginLoader()
loader.discover(local_dirs=[
    "/path/to/my-plugin",
    "/path/to/another-plugin"
])

# Plugins are now available in registry
print(TechniqueRegistry.list())
```

### Testing Plugins

Create comprehensive tests for your plugins:

```python
# tests/test_techniques.py
import pytest
from raglib.schemas import Document
from my_raglib_plugin.techniques import CustomRetriever

def test_custom_retriever():
    """Test custom retriever functionality."""
    retriever = CustomRetriever(similarity_threshold=0.5)
    
    docs = [
        Document(id="1", text="machine learning algorithms"),
        Document(id="2", text="deep learning networks"), 
        Document(id="3", text="cooking recipes")
    ]
    
    result = retriever.process("machine learning", docs)
    
    # Should retrieve relevant documents
    assert len(result.documents) >= 1
    assert result.documents[0].id == "1"
    assert result.metadata["threshold"] == 0.5

def test_custom_retriever_no_matches():
    """Test retriever with no matching documents."""
    retriever = CustomRetriever(similarity_threshold=0.9)
    
    docs = [Document(id="1", text="completely unrelated content")]
    result = retriever.process("machine learning", docs)
    
    assert len(result.documents) == 0
```

## Plugin Discovery API

### PluginLoader Class

```python
from raglib.plugins import PluginLoader

# Initialize loader
loader = PluginLoader()

# Discover all plugins
discovered = loader.discover(
    local_dirs=[],           # Optional local directories
    force_reload=False       # Force re-discovery of entry points
)

print(f"Discovered {discovered['techniques']} techniques")
print(f"Discovered {discovered['adapters']} adapters") 
print(f"Discovered {discovered['architectures']} architectures")
```

### Manual Plugin Registration

Register plugins manually if needed:

```python
from raglib.registry import TechniqueRegistry
from my_plugin.techniques import CustomTechnique

# Register directly
TechniqueRegistry.register("my_custom", CustomTechnique)

# Use immediately
technique = TechniqueRegistry.get("my_custom")
```

### Plugin Information

Get information about loaded plugins:

```python
# List all available techniques
techniques = TechniqueRegistry.list()
print("Available techniques:", techniques)

# Get technique class
technique_class = TechniqueRegistry.get_class("custom_retriever")
print(f"Class: {technique_class.__name__}")
print(f"Module: {technique_class.__module__}")
```

## Advanced Plugin Features

### Configuration Support

Add configuration support to your plugins:

```python
class ConfigurableRetriever(RAGTechnique):
    """Retriever with external configuration support."""
    
    @classmethod
    def from_config(cls, config: dict):
        return cls(
            threshold=config.get("threshold", 0.7),
            max_results=config.get("max_results", 10)
        )
    
    def __init__(self, threshold=0.7, max_results=10):
        super().__init__()
        self.threshold = threshold
        self.max_results = max_results
```

### Plugin Dependencies

Handle optional dependencies gracefully:

```python
class OptionalDependencyTechnique(RAGTechnique):
    """Technique with optional external dependencies."""
    
    def __init__(self):
        super().__init__()
        self._check_dependencies()
    
    def _check_dependencies(self):
        try:
            import optional_library
            self.has_optional = True
        except ImportError:
            self.has_optional = False
            print("Warning: optional_library not found, using fallback")
    
    def process(self, query: str, docs: list[Document]) -> RagResult:
        if self.has_optional:
            return self._advanced_process(query, docs)
        else:
            return self._fallback_process(query, docs)
```

### Lifecycle Hooks

Implement plugin lifecycle methods:

```python
class LifecycleAwareTechnique(RAGTechnique):
    """Technique with initialization and cleanup hooks."""
    
    def initialize(self):
        """Called after plugin discovery."""
        self.setup_resources()
    
    def cleanup(self):
        """Called on shutdown.""" 
        self.release_resources()
    
    def setup_resources(self):
        # Initialize expensive resources
        pass
    
    def release_resources(self):
        # Clean up resources
        pass
```

## Best Practices

### Plugin Design

1. **Single responsibility**: Each plugin should have a focused purpose
2. **Clear interfaces**: Follow RAGLib base class contracts
3. **Error handling**: Handle errors gracefully with informative messages
4. **Documentation**: Include docstrings and usage examples
5. **Testing**: Comprehensive test coverage for all functionality

### Performance

1. **Lazy loading**: Don't load expensive resources until needed
2. **Caching**: Cache expensive computations when appropriate
3. **Memory management**: Clean up resources properly
4. **Batch processing**: Support batch operations when possible

### Compatibility

1. **Version pinning**: Specify compatible RAGLib versions
2. **Python versions**: Test with multiple Python versions
3. **Dependencies**: Minimize required dependencies
4. **Graceful degradation**: Handle missing optional dependencies

### Distribution

1. **Semantic versioning**: Use proper version numbering
2. **Change logs**: Document changes between versions
3. **License**: Include appropriate license
4. **Metadata**: Rich package metadata for discoverability

## Troubleshooting

### Common Issues

**Plugin not discovered**:
```python
# Check entry points
import pkg_resources
for entry_point in pkg_resources.iter_entry_points("raglib.techniques"):
    print(entry_point.name, entry_point.module_name)
```

**Import errors**:
```python
# Verify plugin installation
import my_raglib_plugin
print(my_raglib_plugin.__file__)
```

**Registration conflicts**:
```python
# Check for name collisions
existing = TechniqueRegistry.list()
print("Existing techniques:", existing)
```

**Local plugin not found**:
```python
# Verify directory structure
import os
plugin_dir = "/path/to/plugin" 
print("Files:", os.listdir(plugin_dir))
print("Python files:", [f for f in os.listdir(plugin_dir) if f.endswith('.py')])
```

### Debug Mode

Enable verbose plugin discovery:

```python
loader = PluginLoader(verbose=True)
discovered = loader.discover()
# Prints detailed discovery information
```

For comprehensive examples, see `examples/plugin_example/` and `tests/test_plugins.py`.
