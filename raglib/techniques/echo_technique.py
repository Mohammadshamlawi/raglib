"""Echo technique: returns whatever payload it receives (wrapped as TechniqueResult)."""

from typing import Any, Optional, Dict
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry


@TechniqueRegistry.register
class EchoTechnique(RAGTechnique):
    meta = TechniqueMeta(
        name="echo_technique",
        category="utility",
        description="Echoes the input payload back as TechniqueResult."
    )

    def __init__(self):
        super().__init__(self.meta)

    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Return the first positional argument (or kwargs) as the payload."""
        if args:
            payload = args[0]
        else:
            payload = kwargs if kwargs else None
        return TechniqueResult(success=True, payload=payload, meta={"echoed": True})
