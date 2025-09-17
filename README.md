# RAGLib

[![CI](https://github.com/Mohammadshamlawi/raglib/workflows/CI/badge.svg)](https://github.com/Mohammadshamlawi/raglib/actions)
[![PyPI version](https://badge.fury.io/py/rag-techlib.svg)](https://badge.fury.io/py/rag-techlib)
[![Documentation Status](https://readthedocs.org/projects/rag-techlib/badge/?version=latest)](https://rag-techlib.readthedocs.io/en/latest/?badge=latest)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An extensible, abstract framework for building Retrieval-Augmented Generation (RAG) systems with unlimited technique possibilities.**

RAGLib provides a unified, registry-based architecture that allows you to discover, compose, and extend RAG techniques dynamically. Whether you're prototyping with built-in implementations or integrating cutting-edge research, RAGLib adapts to your needs.

## ✨ Core Philosophy

RAGLib is designed around **abstraction** and **extensibility**:

- **🏗️ Abstract Interface**: All techniques implement the same `RAGTechnique.apply()` contract
- **🔍 Dynamic Discovery**: Find and use techniques through the central `TechniqueRegistry`
- **🧩 Infinite Extensibility**: Add new techniques without modifying core library code
- **🔌 Adapter Pattern**: Swap implementations from fallback to production with zero code changes
- **📦 Plugin Ecosystem**: Distribute and discover techniques as standalone packages

## 🚀 Key Features

- **🔧 Unified API**: Every technique, from chunking to generation, uses the same interface
- **🗂️ Registry System**: Dynamically discover, register, and compose techniques
- **🧩 Modular Architecture**: Mix and match any combination of techniques
- **🔌 Plugin Framework**: Easy extensibility with automatic plugin discovery
- **⚡ Production Ready**: Seamless transition from prototypes to production systems
- **📊 Built-in Benchmarking**: Evaluate and compare technique performance
- **📖 Self-Documenting**: Auto-generated documentation from technique metadata

## 🎯 Quick Start

### Installation

```bash
pip install rag-techlib

# For optional dependencies
pip install rag-techlib[faiss]    # Advanced embeddings and vector search
pip install rag-techlib[llm]      # LLM integration
pip install rag-techlib[dev]      # Development tools
pip install rag-techlib[all]      # Everything
```

### Registry-Driven Usage

The power of RAGLib lies in its registry system - discover and use techniques dynamically:

```python
from raglib.registry import TechniqueRegistry
from raglib.schemas import Document

# Discover available techniques
print("Available techniques:", list(TechniqueRegistry.list().keys()))

# Get techniques by category
chunkers = TechniqueRegistry.find_by_category("chunking")
retrievers = TechniqueRegistry.find_by_category("retrieval")

# Load any technique dynamically
ChunkerClass = TechniqueRegistry.get("fixed_size_chunker")
RetrieverClass = TechniqueRegistry.get("dense_retriever")

# Use with consistent interface
chunker = ChunkerClass(chunk_size=256, overlap=20)
retriever = RetrieverClass()

# All techniques use the same apply() method
documents = [Document(id="1", text="Your content here...")]
chunks_result = chunker.apply(documents)
retrieval_result = retriever.apply(query="search query", corpus=chunks_result.payload)
```

### Configuration-Driven Pipelines

Build entire RAG systems from configuration:

```python
from raglib.pipelines import Pipeline

# Define pipeline from configuration
pipeline_config = [
    ("chunker", "semantic_chunker", {"similarity_threshold": 0.8}),
    ("retriever", "dense_retriever", {"top_k": 5}),
    ("reranker", "mmr", {"lambda_param": 0.7}),
    ("generator", "llm_generator", {"temperature": 0.7})
]

# Build pipeline dynamically
pipeline_steps = []
for name, technique_name, params in pipeline_config:
    TechniqueClass = TechniqueRegistry.get(technique_name)
    technique = TechniqueClass(**params)
    pipeline_steps.append(technique)

pipeline = Pipeline(pipeline_steps)

# Run the complete RAG system
result = pipeline.run("What is machine learning?")
```

## 🧩 Extensible Architecture

### Adding Custom Techniques

RAGLib's registry system makes adding new techniques trivial:

```python
from raglib.core import RAGTechnique, TechniqueMeta, TechniqueResult
from raglib.registry import TechniqueRegistry

@TechniqueRegistry.register
class MyCustomRetriever(RAGTechnique):
    meta = TechniqueMeta(
        name="my_custom_retriever",
        category="retrieval", 
        description="My innovative retrieval technique",
        version="1.0.0"
    )
    
    def __init__(self, custom_param=None):
        super().__init__(self.meta)
        self.custom_param = custom_param
    
    def apply(self, query, corpus, **kwargs):
        # Your custom logic here
        results = self._my_retrieval_logic(query, corpus)
        return TechniqueResult(
            success=True,
            payload={"hits": results}
        )

# Technique is automatically available in registry
retriever = TechniqueRegistry.get("my_custom_retriever")(custom_param="value")
```

### Plugin Development

Create distributable technique packages:

```python
# In your plugin package
from raglib.core import RAGTechnique, TechniqueMeta
from raglib.registry import TechniqueRegistry

@TechniqueRegistry.register
class AdvancedSemanticRetriever(RAGTechnique):
    meta = TechniqueMeta(
        name="advanced_semantic_retriever",
        category="retrieval",
        description="State-of-the-art semantic retrieval",
        dependencies=["transformers", "faiss-gpu"]
    )
    
    def apply(self, query, corpus, **kwargs):
        # Advanced implementation
        pass

# Users can install and use immediately:
# pip install your-raglib-plugin
# from raglib.registry import TechniqueRegistry
# retriever = TechniqueRegistry.get("advanced_semantic_retriever")()
```

## 📚 Built-in Technique Categories

RAGLib provides production-ready implementations across all RAG components:

### 🔨 Document Processing
- **Content-Aware Chunking**: Content-aware chunking that respects text structure and natural boundaries
- **Document-Specific Chunking**: Document-specific chunking that adapts to document type
- **Fixed Size Chunking**: Fixed-size text chunking with overlap support
- **Document-Specific Chunking**: Parent document retrieval with small-to-large chunk mapping
- **Propositional Chunking**: Propositional chunking based on semantic propositions
- **Recursive Chunking**: Recursive chunking with hierarchical text splitting
- **Semantic Chunking**: Semantic similarity-based chunking with configurable embedder
- **Sentence Window Chunking**: Sentence-based windowing with configurable window size and overlap

### 🔍 Information Retrieval  

**Sparse Retrieval Techniques:**
- **BM25**: Classical probabilistic ranking function for sparse retrieval
- **TF-IDF**: Term frequency-inverse document frequency with cosine similarity
- **Lexical Matcher**: Configurable lexical matching (exact, substring, token overlap)
- **SPLADE**: Sparse lexical and dense expansion hybrid approach
- **Lexical Transformer**: Transformer-aware lexical retrieval with attention weighting

**Dense & Hybrid Retrieval:**
- **Dense Retrieval**: Vector-based semantic search with adapter support
- **Hybrid Retrieval**: Combine multiple retrieval strategies

### 🎯 Result Refinement
- **MMR Reranking**: Balance relevance and diversity
- **Cross-Encoder Reranking**: Pairwise relevance scoring
- **Custom Rerankers**: Easily add domain-specific reranking logic

### 💬 Response Generation
- **LLM Generation**: Flexible text generation with adapter support
- **HyDE**: Query expansion using hypothetical document generation
- **Template Generation**: Structured response formatting

### 🔧 System Components
- **Pipeline Orchestration**: Chain techniques with error handling
- **Fusion-in-Decoder**: Advanced multi-context generation
- **Adapter Interfaces**: Seamless integration with external libraries

*All categories support both fallback implementations (for prototyping) and production adapters (for deployment).*

## 🔄 Adapter System

Seamlessly transition from development to production:

```python
# Development: Use built-in fallbacks
retriever = TechniqueRegistry.get("dense_retriever")()  # Uses DummyEmbedder

# Production: Plug in real implementations  
from sentence_transformers import SentenceTransformer
from your_vector_db import ProductionVectorStore

class RealEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed(self, texts):
        return self.model.encode(texts)

retriever = TechniqueRegistry.get("dense_retriever")(
    embedder=RealEmbedder(),
    vectorstore=ProductionVectorStore()
)

# Same interface, production performance
result = retriever.apply(query="search query", corpus=documents)
```

## 📊 Benchmarking & Evaluation

```python
from raglib.benchmark import BenchmarkHarness

# Evaluate any technique
retriever = TechniqueRegistry.get("bm25")()

harness = BenchmarkHarness()
results = harness.run_benchmark(
    technique=retriever,
    dataset_path="evaluation_data.jsonl",
    metrics=["precision", "recall", "f1"]
)

print(f"F1 Score: {results['metrics']['f1']:.3f}")
```

## 📖 Documentation

- **[Getting Started](https://rag-techlib.readthedocs.io/en/latest/getting_started/)** - Installation and first steps
- **[Techniques Guide](https://rag-techlib.readthedocs.io/en/latest/techniques/)** - Complete technique catalog  
- **[API Reference](https://rag-techlib.readthedocs.io/en/latest/api/)** - Detailed API documentation
- **[Plugin Development](https://rag-techlib.readthedocs.io/en/latest/plugins/)** - Creating custom techniques
- **[Benchmarking Guide](https://rag-techlib.readthedocs.io/en/latest/benchmarking/)** - Evaluation framework

## 🛠️ Development

### Setup

```bash
git clone https://github.com/your-org/raglib.git
cd raglib
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]
```

### Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=raglib

# Run only fast tests  
pytest -m "not slow"

# Run by category
pytest -m unit        # Unit tests
pytest -m integration # Integration tests
```

### Code Quality

```bash
# Linux/macOS
make format           # Format code
make lint            # Run linting  
make type-check      # Type checking
make all-checks      # All quality checks

# Windows
.\build.ps1 format
.\build.ps1 lint
.\build.ps1 type-check
```

### Documentation

```bash
# Generate technique index
make docs-generate    # Linux/macOS
.\build.ps1 docs-generate  # Windows

# Build and serve docs
mkdocs serve         # Development server
mkdocs build         # Build static site
```

## 🤝 Contributing

We welcome contributions! RAGLib thrives on community-contributed techniques and improvements.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-technique`)
3. **Add your technique** following our [Contributing Guide](CONTRIBUTING.md)
4. **Include comprehensive tests** 
5. **Ensure all quality checks pass**
6. **Submit a pull request**

### Contribution Types

- **🧬 New Techniques**: Add novel RAG techniques to the registry
- **🔧 Adapters**: Create integrations with external libraries
- **📊 Benchmarks**: Contribute evaluation datasets and metrics
- **📚 Documentation**: Improve guides, examples, and API docs
- **🐛 Bug Fixes**: Help maintain code quality
- **⚡ Performance**: Optimize existing implementations

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🎯 Why RAGLib?

### For Researchers
- **🔬 Rapid Prototyping**: Test new ideas without infrastructure overhead
- **📊 Standardized Evaluation**: Compare techniques fairly with built-in benchmarks  
- **🔄 Easy Reproduction**: Deterministic fallbacks ensure reproducible results
- **📦 Simple Distribution**: Share techniques as plugins

### For Engineers
- **🏗️ Production Ready**: Seamless transition from prototype to production
- **🔧 Flexible Integration**: Adapter pattern works with any external library
- **📈 Scalable Architecture**: Registry system handles growing technique libraries
- **🛡️ Robust Error Handling**: Graceful fallbacks prevent system failures

### For Teams
- **🤝 Collaborative Development**: Registry enables parallel technique development
- **📋 Configuration Management**: Build systems from declarative configs
- **🔍 Discovery**: Find and evaluate techniques without code diving
- **📖 Self-Documenting**: Metadata-driven documentation stays current

## 🌟 Community & Ecosystem

- **GitHub Discussions**: Share ideas and get help
- **Plugin Registry**: Discover community-contributed techniques
- **Research Papers**: Implementations of latest RAG research
- **Industry Examples**: Production-tested technique combinations

## 🚀 Roadmap

- **Multi-modal Support**: Image and audio RAG techniques
- **Distributed Computing**: Scale to massive document collections
- **AutoRAG**: Automatic technique selection and optimization
- **Visual Pipeline Builder**: GUI for technique composition
- **Enterprise Features**: Advanced monitoring and deployment tools

## 📝 License

```
MIT License

Copyright (c) 2025 RAGLib Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📚 Citation

If you use RAGLib in your research, please cite:

```bibtex
@software{raglib,
  title = {RAGLib: An Extensible Framework for Retrieval-Augmented Generation},
  author = {RAGLib Contributors},
  year = {2025},
  url = {https://github.com/Mohammadshamlawi/raglib},
  note = {An abstract, registry-based library for building RAG systems}
}
```

---

**RAGLib** - *Building the future of Retrieval-Augmented Generation, one technique at a time.*

🌐 **Website**: https://rag-techlib.readthedocs.io  
🐛 **Issues**: https://github.com/Mohammadshamlawi/raglib/issues  
💬 **Discussions**: https://github.com/Mohammadshamlawi/raglib/discussions  
📦 **PyPI**: https://pypi.org/project/rag-techlib/