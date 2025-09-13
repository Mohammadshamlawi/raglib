"""Template/stub for a technique file.

Each real technique should live in its own file in this directory and implement the
RAGTechnique.apply(...) method. This file provides a minimal template that can be
copied/extended in later milestones.
"""
from dataclasses import dataclass
from typing import Any
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult

@dataclass
class TemplateMeta(TechniqueMeta):
    pass

class TemplateTechnique(RAGTechnique):
    meta = TechniqueMeta(
        name="template_technique",
        category="placeholder",
        description="A placeholder template for techniques in raglib."
    )

    def __init__(self):
        super().__init__(self.meta)

    def apply(self, *args, **kwargs) -> Any:
        """Placeholder apply method. Real techniques will override this. """
        return TechniqueResult(success=True, payload={"note": "template"})
