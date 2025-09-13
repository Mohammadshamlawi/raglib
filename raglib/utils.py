from typing import Any, Dict, Iterable, Optional, Sequence
from .core import TechniqueResult


def normalize_result(obj: Any) -> Any:
    """If obj is TechniqueResult return its payload, otherwise return obj."""
    if isinstance(obj, TechniqueResult):
        return obj.payload
    return obj


def merge_meta(meta_list: Sequence[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Shallow-merge a sequence of meta dicts into one dict.

    Later entries override earlier ones on key collisions.
    """
    merged: Dict[str, Any] = {}
    for m in meta_list:
        if not m:
            continue
        if not isinstance(m, dict):
            continue
        merged.update(m)
    return merged
