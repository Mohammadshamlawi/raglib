"""
RAGLib - A comprehensive library of RAG techniques with a unified API.

Copyright (c) 2025 RAGLib Contributors
Licensed under the MIT License (see LICENSE file for details).
"""

__all__ = [
    "core",
    "registry",
    "schemas",
    "utils",
    "pipelines",
    "techniques",
    "adapters",
]

# package version
__version__ = "0.1.0"
__author__ = "RAGLib Contributors"
__email__ = "contributors@raglib.org"
__license__ = "MIT"

# Import techniques to ensure they are registered
# This triggers the @TechniqueRegistry.register decorators
from . import techniques  # noqa: F401
