# Contributing to RAGLib

Thank you for your interest in contributing to RAGLib! This document provides comprehensive guidelines and instructions for contributing to the project. We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, code contributions, and more.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Code Style and Standards](#code-style-and-standards) 
- [Testing Guidelines](#testing-guidelines)
- [Adding New Techniques](#adding-new-techniques)
- [Plugin Development](#plugin-development)
- [Running Examples and Documentation](#running-examples-and-documentation)
- [Submitting Contributions](#submitting-contributions)
- [Review Process](#review-process)
- [Community Guidelines](#community-guidelines)

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-org/raglib.git
cd raglib

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install in development mode with all extras
pip install -e .[dev,tests,all]
```

### 2. Verify Installation

```bash
# Run tests to ensure everything works
pytest

# Run quick tests (skip slow integration tests)
pytest -m "not slow"

# Run examples to verify functionality
python examples/quick_start.py
raglib-cli quick-start

# Generate and build documentation
python tools/generate_techniques_index.py
mkdocs build
```

### 3. Development Tools Setup

**Linux/macOS users:**
```bash
# Use make for common tasks
make dev-install    # Install with dev dependencies
make test          # Run tests
make docs          # Build documentation
make all-checks    # Run all code quality checks
```

**Windows users:**
```powershell
# Use PowerShell build script
.\build.ps1 test         # Run tests
.\build.ps1 docs         # Build documentation
.\build.ps1 coverage     # Run tests with coverage
```

```batch
# Or use batch script
.\build.bat test         # Run tests
.\build.bat docs         # Build documentation
```

## Code Style and Standards

### Code Formatting

We use **black**, **isort**, and **ruff** for code formatting and linting:

```bash
# Format code
black raglib/ tests/ examples/
isort raglib/ tests/ examples/

# Lint code
ruff check raglib/ tests/ examples/

# Type checking
mypy raglib/
```

**Automated formatting:**
```bash
make format      # Linux/macOS
.\build.ps1 format  # Windows PowerShell
```

### Code Standards

- **Line length**: 88 characters (black default)
- **Import organization**: Use isort with black profile
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public APIs
- **Variable naming**: Use descriptive names, follow PEP 8

### Example Code Style

```python
"""Module docstring describing the purpose."""

from typing import Dict, List, Optional

from raglib.core import RAGTechnique, TechniqueMeta
from raglib.schemas import RagResult


class ExampleTechnique(RAGTechnique):
    """Example technique demonstrating code style.
    
    This class shows the expected code style including type hints,
    docstrings, and formatting conventions.
    
    Args:
        param1: Description of the first parameter.
        param2: Optional parameter with default value.
    
    Example:
        >>> technique = ExampleTechnique(param1="value")
        >>> result = technique.apply(query="test")
        >>> print(result.answer)
    """
    
    meta = TechniqueMeta(
        name="example_technique",
        description="Example technique for demonstration",
        category="retrieval",
        version="1.0.0",
        dependencies=[]
    )
    
    def __init__(self, param1: str, param2: Optional[int] = None) -> None:
        self.param1 = param1
        self.param2 = param2 or 42
    
    def apply(self, query: str, **kwargs) -> RagResult:
        """Apply the technique to process the query.
        
        Args:
            query: The input query to process.
            **kwargs: Additional keyword arguments.
        
        Returns:
            The processing result.
        
        Raises:
            ValueError: If query is empty.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Implementation here
        return RagResult(answer=f"Processed: {query}")
```

## Testing Guidelines

### Test Structure

RAGLib uses pytest for testing with the following structure:

```
tests/
├── test_adapters.py           # Adapter interface tests
├── test_techniques/           # Individual technique tests
├── test_pipelines.py         # Pipeline composition tests
├── test_examples.py          # Example script smoke tests
└── conftest.py               # Shared test fixtures
```

### Writing Tests

**Test Categories:**
- **Unit tests**: Fast tests of individual components (`@pytest.mark.unit`)
- **Integration tests**: Multi-component interactions (`@pytest.mark.integration`)
- **Slow tests**: Long-running tests (`@pytest.mark.slow`)
- **CLI tests**: Command-line interface tests (`@pytest.mark.cli`)

**Example test:**
```python
import pytest
from raglib.techniques.example import ExampleTechnique
from raglib.schemas import Document


class TestExampleTechnique:
    """Test suite for ExampleTechnique."""
    
    @pytest.fixture
    def technique(self):
        """Create technique instance for testing."""
        return ExampleTechnique(param1="test")
    
    @pytest.fixture  
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(id="1", text="Sample document one"),
            Document(id="2", text="Sample document two")
        ]
    
    def test_apply_basic(self, technique, sample_documents):
        """Test basic functionality."""
        result = technique.apply(
            query="test query",
            corpus=sample_documents
        )
        
        assert result.answer is not None
        assert len(result.payload.get("hits", [])) > 0
    
    def test_apply_empty_query(self, technique):
        """Test error handling for empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            technique.apply(query="")
    
    @pytest.mark.slow
    def test_apply_large_corpus(self, technique):
        """Test with large corpus (marked as slow)."""
        large_corpus = [
            Document(id=str(i), text=f"Document {i}")
            for i in range(1000)
        ]
        
        result = technique.apply(
            query="test",
            corpus=large_corpus
        )
        
        assert result is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run fast tests only
pytest -m "not slow"

# Run tests with coverage
pytest --cov=raglib --cov-report=html

# Run specific test file
pytest tests/test_techniques/test_example.py

# Run tests matching pattern
pytest -k "test_apply"

# Verbose output
pytest -v
```

### Test Requirements

- **All new code must have tests** with >90% coverage
- **Tests must be fast** (<1s each, mark slow tests with `@pytest.mark.slow`)
- **Tests must be deterministic** (no random failures)
- **Mock external dependencies** (APIs, file systems, etc.)
- **Test edge cases** (empty inputs, errors, boundary conditions)

## Adding New Techniques

### Step-by-Step Process

#### 1. Create the Technique Class

```python
# raglib/techniques/my_new_technique.py

from typing import Dict, Any
from raglib.core import RAGTechnique, TechniqueMeta, TechniqueRegistry
from raglib.schemas import RagResult


@TechniqueRegistry.register
class MyNewTechnique(RAGTechnique):
    """My new RAG technique.
    
    Detailed description of what this technique does,
    when to use it, and how it works.
    
    Args:
        param1: Description of parameter.
        param2: Optional parameter.
    
    Example:
        >>> technique = MyNewTechnique(param1="value")
        >>> result = technique.apply(query="test", corpus=docs)
        >>> print(result.answer)
    """
    
    meta = TechniqueMeta(
        name="my_new_technique",
        description="Brief description for documentation",
        category="retrieval",  # or "chunking", "reranking", "generation"
        version="1.0.0",
        dependencies=["optional-package"]  # or [] if none
    )
    
    def __init__(self, param1: str, param2: int = 42):
        self.param1 = param1
        self.param2 = param2
    
    def apply(self, query: str, **kwargs) -> RagResult:
        """Apply the technique.
        
        Args:
            query: Input query.
            **kwargs: Additional parameters (corpus, top_k, etc.)
        
        Returns:
            Result with answer and metadata.
        """
        # Your implementation here
        return RagResult(
            answer=f"Result for: {query}",
            payload={"technique": "my_new_technique"}
        )
```

#### 2. Register the Technique

Add import to `raglib/techniques/__init__.py`:

```python
from .my_new_technique import MyNewTechnique

__all__ = [
    # ... existing techniques
    "MyNewTechnique",
]
```

#### 3. Write Comprehensive Tests

```python
# tests/test_my_new_technique.py

import pytest
from raglib.techniques.my_new_technique import MyNewTechnique


class TestMyNewTechnique:
    """Test suite for MyNewTechnique."""
    
    def test_basic_functionality(self):
        """Test basic technique functionality."""
        technique = MyNewTechnique(param1="test")
        result = technique.apply(query="sample query")
        
        assert result.answer is not None
        assert "technique" in result.payload
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            MyNewTechnique(param1="")  # Invalid parameter
    
    # Add more tests...
```

#### 4. Add Documentation

Create or update documentation in `docs/api/techniques.md` or add inline docstring examples.

#### 5. Update Documentation Index

```bash
# Regenerate techniques documentation
python tools/generate_techniques_index.py

# Or use build scripts
make docs-generate  # Linux/macOS
.\build.ps1 docs-generate  # Windows
```

#### 6. Create Example Usage

Add example to `examples/` directory or update existing examples:

```python
# examples/my_new_technique_example.py

from raglib.techniques.my_new_technique import MyNewTechnique
from raglib.schemas import Document


def main():
    """Demonstrate MyNewTechnique usage."""
    # Create sample documents
    docs = [
        Document(id="1", text="Sample document content"),
        Document(id="2", text="Another document")
    ]
    
    # Initialize technique
    technique = MyNewTechnique(param1="example")
    
    # Apply technique
    result = technique.apply(
        query="What is this about?",
        corpus=docs,
        top_k=5
    )
    
    print(f"Answer: {result.answer}")
    print(f"Metadata: {result.payload}")


if __name__ == "__main__":
    main()
```

### Technique Guidelines

- **Follow the RAGTechnique interface** exactly
- **Use TechniqueMeta** for complete metadata
- **Register with @TechniqueRegistry.register**
- **Handle edge cases** gracefully (empty inputs, missing data)
- **Include comprehensive docstrings** with examples
- **Support common parameters** (top_k, corpus, etc.)
- **Return proper RagResult** objects
- **Consider adapter pattern** for external dependencies

## Plugin Development

RAGLib supports a plugin architecture that allows third-party packages to extend the library with custom techniques, adapters, and architectures. This section covers how to create and distribute RAGLib plugins.

### Plugin Entry Points

RAGLib uses Python entry points to discover and load plugins automatically. Plugins can register components in three categories:

- **`raglib.techniques`** - Custom RAG techniques
- **`raglib.adapters`** - Custom adapters (embedders, vector stores, etc.)
- **`raglib.architectures`** - Custom RAG architectures

### Creating a Plugin Package

#### 1. Package Structure

Create a standard Python package with the following structure:

```
my-raglib-plugin/
├── setup.py or pyproject.toml
├── README.md
├── my_raglib_plugin/
│   ├── __init__.py
│   ├── techniques/
│   │   ├── __init__.py
│   │   └── my_technique.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── my_adapter.py
│   └── architectures/
│       ├── __init__.py
│       └── my_architecture.py
└── tests/
    └── test_plugin.py
```

#### 2. Define Entry Points in pyproject.toml

```toml
[project]
name = "my-raglib-plugin"
version = "1.0.0"
description = "Custom RAGLib plugin with specialized techniques"
dependencies = [
    "raglib>=0.1.0",
    # Add your specific dependencies here
]

[project.entry-points."raglib.techniques"]
my_custom_technique = "my_raglib_plugin.techniques.my_technique:MyCustomTechnique"
advanced_retriever = "my_raglib_plugin.techniques.advanced:AdvancedRetriever"

[project.entry-points."raglib.adapters"]
my_embedder = "my_raglib_plugin.adapters.my_adapter:MyEmbedder"
custom_vectorstore = "my_raglib_plugin.adapters.vectorstore:CustomVectorStore"

[project.entry-points."raglib.architectures"]
my_architecture = "my_raglib_plugin.architectures.my_architecture:MyArchitecture"
```

#### 3. Implement Plugin Components

**Technique Example:**
```python
# my_raglib_plugin/techniques/my_technique.py
from raglib.core import RAGTechnique, TechniqueMeta
from raglib.schemas import RagResult, Document
from raglib.registry import TechniqueRegistry
from typing import List, Dict, Any

@TechniqueRegistry.register("my_custom_technique")
class MyCustomTechnique(RAGTechnique):
    """A custom RAG technique for specialized retrieval."""
    
    meta = TechniqueMeta(
        name="My Custom Technique",
        description="A specialized technique for custom use cases",
        author="Your Name",
        email="your.email@example.com",
        version="1.0.0",
        category="retrieval",
        tags=["custom", "specialized", "retrieval"],
        requirements=["my-special-dependency>=1.0.0"]
    )
    
    def __init__(self, custom_param: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def apply(
        self, 
        query: str, 
        corpus: List[Document], 
        top_k: int = 5,
        **kwargs
    ) -> RagResult:
        """Apply the custom technique."""
        # Implement your custom logic here
        results = self._custom_retrieval_logic(query, corpus, top_k)
        
        return RagResult(
            query=query,
            retrieved_documents=results,
            metadata={
                "technique": "my_custom_technique",
                "custom_param": self.custom_param,
                **kwargs
            }
        )
    
    def _custom_retrieval_logic(
        self, 
        query: str, 
        corpus: List[Document], 
        top_k: int
    ) -> List[Document]:
        """Implement your custom retrieval logic."""
        # Your custom implementation here
        pass
```

**Adapter Example:**
```python
# my_raglib_plugin/adapters/my_adapter.py
from raglib.adapters.base import BaseEmbedder
from typing import List
import numpy as np

class MyEmbedder(BaseEmbedder):
    """Custom embedder using a specialized model."""
    
    def __init__(self, model_name: str = "my-custom-model"):
        self.model_name = model_name
        # Initialize your custom embedding model
        self._load_model()
    
    def _load_model(self):
        """Load your custom embedding model."""
        # Implementation specific to your embedding approach
        pass
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using the custom model."""
        # Your custom embedding logic
        embeddings = []
        for text in texts:
            embedding = self._embed_single_text(text)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self._embed_single_text(query)
    
    def _embed_single_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        # Your custom embedding implementation
        pass
```

#### 4. Plugin Discovery and Usage

Once installed, RAGLib automatically discovers and loads your plugin components:

```python
from raglib import TechniqueRegistry

# Your plugin technique is now available
technique = TechniqueRegistry.get("my_custom_technique")

# Use it like any other RAGLib technique
result = technique.apply(query="test query", corpus=documents)
```

### Plugin Best Practices

#### Code Quality
- **Follow RAGLib conventions** - Use the same code style and patterns as the core library
- **Comprehensive testing** - Include unit tests and integration tests
- **Type hints** - Use proper type annotations throughout your code
- **Documentation** - Provide clear docstrings and usage examples

#### Compatibility
- **Version compatibility** - Specify minimum RAGLib version requirements
- **Dependency management** - Use optional dependencies for heavy libraries
- **Graceful degradation** - Handle missing dependencies gracefully
- **Cross-platform support** - Test on multiple operating systems

#### Performance
- **Lazy loading** - Only load heavy models when needed
- **Memory efficiency** - Clean up resources properly
- **Batch processing** - Support batch operations where possible
- **Caching** - Implement appropriate caching strategies

#### Distribution
- **Clear naming** - Use descriptive package names (e.g., "raglib-scientific-papers")
- **Semantic versioning** - Follow semantic versioning for releases
- **License compatibility** - Ensure license compatibility with RAGLib (MIT)
- **Documentation** - Provide comprehensive README and examples

### Plugin Testing

#### Local Development
```bash
# Install your plugin in development mode
pip install -e /path/to/your/plugin

# Test that RAGLib can discover it
python -c "from raglib import TechniqueRegistry; print(TechniqueRegistry.list())"

# Run your plugin tests
pytest tests/
```

#### Integration Testing
```python
# tests/test_plugin.py
import pytest
from raglib import TechniqueRegistry
from raglib.schemas import Document

def test_plugin_registration():
    """Test that the plugin is properly registered."""
    assert "my_custom_technique" in TechniqueRegistry.list()

def test_plugin_functionality():
    """Test that the plugin works correctly."""
    technique = TechniqueRegistry.get("my_custom_technique")
    
    documents = [
        Document(id="1", text="Sample document 1"),
        Document(id="2", text="Sample document 2"),
    ]
    
    result = technique.apply("test query", documents)
    
    assert result.query == "test query"
    assert len(result.retrieved_documents) <= len(documents)
    assert "my_custom_technique" in result.metadata["technique"]
```

### Plugin Examples

#### Academic Research Plugin
```toml
# Example: raglib-academic
[project.entry-points."raglib.techniques"]
citation_retrieval = "raglib_academic.techniques:CitationRetrieval"
paper_similarity = "raglib_academic.techniques:PaperSimilarity"
author_disambiguation = "raglib_academic.techniques:AuthorDisambiguation"
```

#### Domain-Specific Plugin
```toml
# Example: raglib-medical
[project.entry-points."raglib.techniques"]
medical_ner_retrieval = "raglib_medical.techniques:MedicalNERRetrieval"
symptom_matching = "raglib_medical.techniques:SymptomMatching"
drug_interaction = "raglib_medical.techniques:DrugInteractionRetrieval"
```

#### Adapter Plugin
```toml
# Example: raglib-enterprise
[project.entry-points."raglib.adapters"]
enterprise_search = "raglib_enterprise.adapters:EnterpriseSearchAdapter"
secure_embedder = "raglib_enterprise.adapters:SecureEmbedder"
audit_logger = "raglib_enterprise.adapters:AuditLogger"
```

### Publishing Your Plugin

1. **Package your plugin** using standard Python packaging tools
2. **Test thoroughly** across different Python versions and platforms
3. **Write documentation** including installation and usage instructions
4. **Publish to PyPI** with a clear name like `raglib-your-domain`
5. **Submit to community registry** (if available) for discoverability

### Community Plugins

RAGLib maintains a registry of community plugins. To have your plugin listed:

1. Ensure it follows all best practices
2. Include comprehensive documentation and examples
3. Provide reliable maintenance and support
4. Submit a PR to add your plugin to the community registry

## Running Examples and Documentation

### Examples

```bash
# Run individual examples
python examples/quick_start.py
python examples/e2e_toy.py
python examples/benchmark_run.py

# Test examples work correctly
pytest tests/test_examples.py

# Use CLI interface
raglib-cli quick-start
raglib-cli run-example quick_start
raglib-cli docs-build
```

### Documentation

#### Building Documentation

```bash
# Generate techniques index (automatic in build)
python tools/generate_techniques_index.py

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve  # Visit http://localhost:8000
```

#### Cross-Platform Documentation Building

**Linux/macOS:**
```bash
make docs           # Build documentation
make docs-serve     # Build and serve locally
make docs-generate  # Generate techniques index only
```

**Windows:**
```powershell
.\build.ps1 docs           # Build documentation
.\build.ps1 docs-serve     # Build and serve locally
.\build.ps1 docs-generate  # Generate techniques index only
```

```batch
.\build.bat docs           # Build documentation  
.\build.bat docs-serve     # Build and serve locally
.\build.bat docs-generate  # Generate techniques index only
```

#### Documentation Structure

```
docs/
├── index.md                 # Homepage
├── getting_started.md       # Getting started guide
├── techniques.md           # Techniques catalog (includes generated)
├── techniques_generated.md # Generated techniques list
├── core_concepts.md        # Core concepts explanation
├── api.md                  # API reference
└── api/
    ├── techniques.md       # Techniques API docs
    └── adapters.md         # Adapters API docs
```

#### Writing Documentation

- **Use clear, concise language**
- **Include code examples** for all features
- **Add cross-references** between related sections
- **Update generated content** by running build scripts
- **Test documentation builds** before submitting PRs

## Submitting Contributions

### Before You Submit

1. **Run all checks:**
   ```bash
   # Code quality
   make all-checks  # Linux/macOS
   .\build.ps1 all-checks  # Windows
   
   # Tests
   make test
   .\build.ps1 test
   
   # Documentation
   make docs
   .\build.ps1 docs
   ```

2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update CHANGELOG.md** with your changes

### Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** following the guidelines above

3. **Commit with clear messages:**
   ```bash
   git commit -m "Add: New semantic chunking technique
   
   - Implements similarity-based document chunking
   - Includes comprehensive tests and documentation  
   - Adds example usage in examples/semantic_chunking.py"
   ```

4. **Push to your fork** and create a Pull Request

5. **Fill out the PR template** completely

### PR Requirements

- [ ] **Tests pass** on all platforms
- [ ] **Code follows style guidelines**
- [ ] **Documentation is updated**  
- [ ] **CHANGELOG.md is updated**
- [ ] **Examples work correctly**
- [ ] **No breaking changes** (or clearly marked)

### Commit Message Format

Use clear, descriptive commit messages:

```
Type: Brief description (50 chars max)

Longer description explaining what and why, not how.
Reference any relevant issues with #123.

- List specific changes
- Include breaking changes if any
- Mention new dependencies
```

**Types:** `Add`, `Fix`, `Update`, `Remove`, `Refactor`, `Docs`, `Test`, `Style`

## Review Process

### What to Expect

1. **Automated checks** run on your PR (tests, linting, etc.)
2. **Maintainer review** within 48-72 hours
3. **Feedback and discussion** if changes are needed
4. **Approval and merge** once requirements are met

### Review Criteria

- **Functionality**: Does it work as intended?
- **Tests**: Adequate test coverage and quality?
- **Documentation**: Clear and complete?
- **Code Quality**: Follows project standards?
- **Compatibility**: Works across platforms?
- **Performance**: No significant regressions?

## Community Guidelines

### Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Getting Help

- **Documentation**: Check the docs first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Discord/Slack**: Join our community chat (if available)

### Types of Contributions

We welcome all types of contributions:

- **Bug reports** and feature requests
- **Code contributions** (techniques, adapters, utilities)
- **Documentation** improvements and translations
- **Examples** and tutorials
- **Testing** and quality assurance
- **Community support** and issue triage

### Recognition

Contributors are recognized in:
- **CHANGELOG.md** for each release
- **README.md** contributors section
- **Documentation** acknowledgments
- **Release notes** for significant contributions

## Entry Points for Third-Party Plugins

RAGLib supports third-party technique plugins via entry points:

```toml
# In your plugin package's pyproject.toml
[project.entry-points."raglib.plugins"]
my_plugin_technique = "my_package.technique:MyTechnique"
```

## Release Process

See [RELEASE.md](RELEASE.md) for detailed release instructions.

## Questions?

If you have questions about contributing:

1. Check this guide and the documentation
2. Search existing GitHub issues
3. Open a new issue with the "question" label
4. Join our community discussions

Thank you for contributing to RAGLib! 🚀
