from collections.abc import Sequence
from typing import Any

from .core import RAGTechnique, TechniqueResult
from .utils import merge_meta


class Pipeline:
    """Simple pipeline runner that composes RAGTechnique instances.

    Each step's `apply` receives the previous step's payload (or the original input)
    and must follow the `RAGTechnique.apply(*args, **kwargs)` signature.
    """

    def __init__(self, steps: Sequence[RAGTechnique]):
        if not isinstance(steps, Sequence):
            raise TypeError("steps must be a sequence of RAGTechnique instances")
        # quick validation (duck-typed)
        for s in steps:
            if not hasattr(s, "apply"):
                raise TypeError("Each step must be a RAGTechnique-like object with an apply method")
        self.steps = list(steps)

    def run(
        self,
        input_data: Any,
        *,
        strict: bool = False,
        return_payload_only: bool = True
    ) -> Any:
        """Run the pipeline.

        Args:
            input_data: initial input passed to the first step.
            strict: if True, raise RuntimeError when a step returns None (or a falsy
                    value that is not an empty container).
            return_payload_only: if True return the final payload; otherwise return
                    {"payload": final_payload, "meta": merged_meta}.
        """
        data = input_data
        metas = []
        for idx, step in enumerate(self.steps):
            result = step.apply(data)
            # capture meta if it's a TechniqueResult
            if isinstance(result, TechniqueResult):
                metas.append(result.meta)
                data = result.payload
            else:
                data = result
            # Strictness: treat None as failure; also treat False/int(0)/"" as failure,
            # but allow empty list/dict
            if strict:
                if data is None:
                    raise RuntimeError(f"Pipeline step {idx} ({getattr(step, 'meta', None)}) returned None")
                if data is False or (isinstance(data, (int, float, str)) and data == ""):
                    raise RuntimeError(f"Pipeline step {idx} ({getattr(step, 'meta', None)}) returned falsy result")
        if return_payload_only:
            return data
        merged = merge_meta(metas)
        return {"payload": data, "meta": merged}
