# RAGLib Testing and Coverage Guide

## Overview

This document provides comprehensive instructions for running tests and generating coverage reports for the RAGLib project.

## Quick Start

### Running Tests

```bash
# Run all tests
make test

# Run tests excluding slow tests
make test-fast  

# Run tests with verbose output
make test-verbose

# Generate coverage report
make coverage
```

### Using pytest directly

```bash
# All tests
pytest

# Skip slow tests 
pytest -m "not slow"

# Verbose output
pytest -v

# With coverage
pytest --cov=raglib --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m cli           # CLI tests only  
pytest -m examples      # Example script tests
```

## Test Categories

The test suite includes several marker categories:

- **unit**: Fast tests of individual components
- **integration**: Tests of component interactions  
- **cli**: Tests of command-line interface functionality
- **examples**: Tests that run example scripts
- **slow**: Long-running integration tests

## Test Structure

```
tests/
├── test_adapters.py              # Adapter interface tests
├── test_bm25_simple.py          # BM25 retrieval tests
├── test_chunking_*.py           # Chunking technique tests
├── test_core.py                 # Core framework tests
├── test_crossencoder_rerank.py  # Cross-encoder reranking tests
├── test_dummy_dense.py          # Dense retrieval tests
├── test_examples.py             # Example script smoke tests
├── test_fid_pipeline.py         # FID pipeline tests  
├── test_generator.py            # Text generation tests
├── test_hyde.py                 # HyDE technique tests
├── test_mmr.py                  # MMR diversification tests
├── test_pipeline.py             # Pipeline composition tests
├── test_production_*.py         # Production usage tests
└── test_registry.py             # Technique registration tests
```

## Coverage Reports

Coverage reports are generated in both terminal and HTML formats:

### Terminal Coverage
```bash
pytest --cov=raglib --cov-report=term-missing
```

Shows coverage percentages and lists uncovered lines.

### HTML Coverage Report
```bash
pytest --cov=raglib --cov-report=html
open htmlcov/index.html  # View in browser
```

Provides detailed line-by-line coverage visualization.

## Configuration

### pytest.ini
The project includes comprehensive pytest configuration:
- Test discovery patterns
- Custom markers  
- Coverage settings
- Output formatting

### pyproject.toml
Advanced configuration in pyproject.toml includes:
- pytest options and markers
- Coverage source paths and exclusions  
- Code quality tool settings (black, isort, mypy)

## Development Workflow

### Setup Development Environment
```bash
make dev-install    # Install with dev dependencies
make all-checks     # Run format, lint, type-check
make test          # Run test suite
```

### Code Quality Checks
```bash
make format        # Format with black and isort
make lint          # Run linting with ruff
make type-check    # Run type checking with mypy
make all-checks    # Run all quality checks
```

### Continuous Integration

The project includes GitHub Actions workflows that:
- Run tests on multiple Python versions (3.8-3.11)
- Run tests on multiple platforms (Linux, Windows, macOS) 
- Generate coverage reports
- Enforce code quality standards
- Build documentation

## Example Test Scenarios

### Running Quick Tests
```bash
# Fast development loop - skip slow integration tests
pytest -m "not slow" -x --tb=short
```

### Full Test Suite
```bash  
# Complete test run with coverage
pytest --cov=raglib --cov-report=html --cov-report=term-missing
```

### Debugging Failed Tests
```bash
# Verbose output with full tracebacks
pytest -vvv --tb=long --no-header
```

### Testing Specific Components
```bash
# Test specific technique
pytest tests/test_bm25_simple.py -v

# Test all chunking techniques
pytest tests/test_chunking_*.py

# Test CLI functionality only
pytest -m cli
```

## Common Issues and Solutions

### Unicode Issues on Windows
Some example scripts use emoji characters. If tests fail with UnicodeEncodeError:
- Tests handle this gracefully with PYTHONIOENCODING=utf-8
- This is expected behavior on Windows with CP-1252 encoding

### Missing Optional Dependencies  
Many techniques work with fallback implementations when optional dependencies are missing:
- FAISS not installed: Uses in-memory vector store
- Transformers not available: Uses dummy embeddings
- OpenAI not configured: Uses deterministic fallbacks

### Slow Test Performance
Use test markers to control test execution:
```bash
pytest -m "not slow"           # Skip slow integration tests
pytest -m "unit"               # Run only fast unit tests  
pytest --maxfail=3             # Stop after 3 failures
pytest -x                      # Stop on first failure
```

## Contributing

When adding new tests:

1. **Use appropriate markers**: Mark slow tests with `@pytest.mark.slow`
2. **Follow naming conventions**: Test files start with `test_`, functions with `test_`
3. **Add docstrings**: Describe what each test validates
4. **Test edge cases**: Include error conditions and boundary cases
5. **Maintain coverage**: Aim for high code coverage on new components

## Advanced Usage

### Custom Test Selection
```bash
# Run tests matching pattern
pytest -k "bm25 or chunking"

# Run tests in specific file matching pattern  
pytest tests/test_pipeline.py -k "echo"

# Combine markers and patterns
pytest -m "unit and not slow" -k "retriever"
```

### Parallel Test Execution
```bash
# Install pytest-xdist for parallel execution
pip install pytest-xdist

# Run tests in parallel
pytest -n auto  # Use all CPU cores
pytest -n 4     # Use 4 processes
```

### Test Data and Fixtures
The test suite uses pytest fixtures for:
- Sample documents and chunks
- Mock adapters and techniques
- Temporary files and directories
- Test configuration

## Makefile Reference

The included Makefile provides convenient shortcuts:

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-fast` | Run tests excluding slow tests |  
| `make test-verbose` | Run tests with verbose output |
| `make coverage` | Run tests with coverage report |
| `make docs` | Build documentation |
| `make docs-serve` | Build and serve docs locally |
| `make format` | Format code with black/isort |
| `make lint` | Run linting with ruff |
| `make type-check` | Run type checking with mypy |
| `make all-checks` | Run all code quality checks |
| `make build` | Build package for distribution |
| `make clean` | Clean build artifacts |

This comprehensive testing setup ensures code quality, reliability, and maintainability throughout the development process.
