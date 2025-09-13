"""Null technique: returns None to simulate a failing/empty step."""

from typing import Any
from ..core import RAGTechnique, TechniqueMeta
# Intentionally not registered — used directly by tests

class NullTechnique(RAGTechnique):
    meta = TechniqueMeta(
        name="null_technique",
        category="test",
        description="A technique that returns None for testing strict pipeline behavior."
    )

    def __init__(self):
        super().__init__(self.meta)

    def apply(self, *args, **kwargs) -> Any:
        """Return None (simulate failure)."""
        return None
