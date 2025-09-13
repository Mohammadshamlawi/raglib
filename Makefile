# RAGLib Project Makefile
# Usage: make [target]

.PHONY: help install dev-install test test-fast test-verbose coverage clean docs docs-serve build upload format lint type-check all-checks

# Default target
help:
	@echo "RAGLib Project Management"
	@echo "========================"
	@echo ""
	@echo "Development Commands:"
	@echo "  install      Install package in development mode"
	@echo "  dev-install  Install package with all dev dependencies"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run tests excluding slow tests"
	@echo "  test-verbose Run tests with verbose output"
	@echo "  coverage     Run tests with coverage report"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs-generate Generate techniques index from registry"
	@echo "  docs         Build documentation (includes index generation)"
	@echo "  docs-serve   Build and serve documentation locally"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run linting with flake8"
	@echo "  type-check   Run type checking with mypy"
	@echo "  all-checks   Run format, lint, and type-check"
	@echo ""
	@echo "Build Commands:"
	@echo "  build        Build package for distribution"
	@echo "  clean        Clean build artifacts"
	@echo "  upload       Upload to PyPI (requires auth)"

# Installation targets
install:
	pip install -e .

dev-install:
	pip install -e ".[dev,tests,all]"

# Testing targets
test:
	pytest

test-fast:
	pytest -m "not slow"

test-verbose:
	pytest -v

coverage:
	pytest --cov=raglib --cov-report=html --cov-report=term-missing

# Documentation targets
docs: docs-generate
	mkdocs build

docs-serve: docs-generate
	mkdocs serve

docs-generate:
	@echo "Generating techniques index..."
	python tools/generate_techniques_index.py

# Code quality targets
format:
	@echo "Formatting code..."
	black raglib tests examples
	isort raglib tests examples
	@echo "Code formatted!"

lint:
	@echo "Running linting..."
	flake8 raglib tests examples
	@echo "Linting passed!"

type-check:
	@echo "Running type checking..."
	mypy raglib
	@echo "Type checking passed!"

all-checks: format lint type-check
	@echo "All code quality checks passed!"

# Build targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Development workflow
dev-setup: clean dev-install all-checks test
	@echo "Development environment ready!"

# CI/CD workflow
ci-check: all-checks test coverage
	@echo "CI checks passed!"
