# Changelog

All notable changes to RAGLib will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and core framework
- Comprehensive documentation automation system
- Cross-platform build scripts (Makefile, PowerShell, Batch)
- Extensive testing infrastructure with pytest
- Community guidelines and contribution workflows

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2024-12-09

### Added
- **Core Framework**
  - Unified RAGTechnique API with consistent `apply()` interface
  - TechniqueRegistry for managing and discovering techniques
  - Document and TechniqueResult schemas for standardized data flow
  - Modular adapter system for embedders, vector stores, and LLMs

- **Built-in Techniques** (15 techniques across 4 categories)
  - **Chunking**: fixed_size_chunker, sentence_window_chunker, semantic_chunker
  - **Retrieval**: bm25_simple, bm25_production, dense_retriever, dummy_dense
  - **Reranking**: mmr_reranker, crossencoder_reranker
  - **Generation**: llm_generator, hyde_generator
  - **Utilities**: echo_technique, null_technique, demo_fixed_chunker

- **Production Features**
  - Pipeline orchestration with RagPipeline class
  - Fallback implementations for all optional dependencies
  - CLI interface via `raglib-cli` command
  - Plugin system with entry point discovery
  - Comprehensive benchmarking framework with metrics

- **Development Tools**
  - pytest-based testing infrastructure with 70%+ coverage
  - Code quality tools (Black, Ruff, MyPy)
  - Automated documentation generation with MkDocs
  - CI/CD workflows for testing and release automation
  - Cross-platform build scripts

- **Documentation**
  - Complete API documentation
  - Getting started guide with examples
  - Techniques reference with generation
  - Plugin development guide
  - Benchmarking tutorial

### Technical Details
- **Python Support**: 3.9, 3.10, 3.11, 3.12
- **Optional Dependencies**: faiss, transformers, openai, sentence-transformers
- **Package Structure**: Modern pyproject.toml with entry points
- **License**: MIT License with full open source compliance
- Core RAGTechnique interface and registry system
- Basic technique implementations:
  - Fixed-size chunking
  - Semantic chunking  
  - Sentence window chunking
  - BM25 retrieval
  - Dense retrieval
  - Cross-encoder reranking
  - MMR diversification
  - HyDE query expansion
  - LLM generation
- Adapter interfaces for embedders, vector stores, and LLMs
- Pipeline composition system
- CLI interface with example commands
- Comprehensive documentation with MkDocs
- Techniques index generation
- Example scripts and tutorials
- Testing suite with >95% coverage
- Cross-platform development tools

### Dependencies
- Core: numpy, typing-extensions
- Optional: faiss-cpu, transformers, openai, sentence-transformers
- Development: pytest, black, ruff, mypy, mkdocs

---

## Release Notes Format

Each release should include:

### Added
- New features and capabilities
- New techniques, adapters, or utilities
- Documentation improvements
- New examples or tutorials

### Changed  
- Modifications to existing functionality
- Performance improvements
- API changes (with migration notes)
- Documentation updates

### Deprecated
- Features marked for removal in future versions
- Migration path to new alternatives

### Removed
- Features removed in this version
- Breaking changes with migration notes

### Fixed
- Bug fixes and corrections
- Performance fixes
- Documentation fixes

### Security
- Security vulnerabilities addressed
- Security improvements

---

## Contributing to Changelog

When contributing:

1. **Add entries to [Unreleased]** section
2. **Use semantic versioning** for version numbers
3. **Include links** to relevant issues/PRs
4. **Group similar changes** together
5. **Be descriptive** but concise
6. **Follow the format** consistently

Example entry:
```markdown
### Added
- New semantic chunking technique ([#123](https://github.com/org/raglib/pull/123))
  - Implements similarity-based document segmentation
  - Supports configurable similarity thresholds
  - Includes comprehensive tests and documentation

### Fixed
- Fixed memory leak in dense retrieval ([#124](https://github.com/org/raglib/issues/124))
- Corrected type hints in adapter interfaces ([#125](https://github.com/org/raglib/pull/125))
```

## Version Numbering

RAGLib follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: New functionality, backward compatible
- **PATCH** version: Bug fixes, backward compatible

### Release Types

- **Major Release** (x.0.0): Significant changes, potential breaking changes
- **Minor Release** (x.y.0): New features, fully backward compatible
- **Patch Release** (x.y.z): Bug fixes and small improvements
- **Pre-release** (x.y.z-alpha/beta/rc): Development versions

### Breaking Changes

When introducing breaking changes:

1. **Deprecate** the old API in a minor release
2. **Document** migration path clearly
3. **Provide** examples of new usage
4. **Remove** deprecated features in next major release

Example:
```markdown
### Deprecated
- `old_function()` is deprecated, use `new_function()` instead
  - Migration: Replace `old_function(x)` with `new_function(param=x)`
  - Will be removed in v2.0.0
```
