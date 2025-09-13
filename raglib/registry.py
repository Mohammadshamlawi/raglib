from typing import Dict, Type, Optional
from .core import RAGTechnique, TechniqueMeta


class TechniqueRegistry:
    """Registry for RAGTechnique classes.

    Register technique classes with the `@TechniqueRegistry.register` decorator.
    """
    _registry: Dict[str, Type[RAGTechnique]] = {}

    @classmethod
    def register(cls, klass: Type[RAGTechnique]) -> Type[RAGTechnique]:
        """Class decorator to register a technique class.

        The technique class must provide a class-level `meta` attribute (preferably
        an instance of TechniqueMeta) with a `name` and `category`.
        """
        meta = getattr(klass, "meta", None)
        if meta is None:
            raise ValueError("Technique class must define a 'meta' attribute.")
        # best-effort validation
        name: Optional[str] = None
        category: Optional[str] = None
        if isinstance(meta, TechniqueMeta):
            name = meta.name
            category = meta.category
        else:
            # duck-typed check
            name = getattr(meta, "name", None)
            category = getattr(meta, "category", None)
        if not name:
            raise ValueError("Technique 'meta' must have a 'name' attribute.")
        cls._registry[name] = klass
        return klass

    @classmethod
    def get(cls, name: str) -> Type[RAGTechnique]:
        return cls._registry[name]

    @classmethod
    def list(cls) -> Dict[str, Type[RAGTechnique]]:
        return dict(cls._registry)

    @classmethod
    def find_by_category(cls, category: str) -> Dict[str, Type[RAGTechnique]]:
        return {
            name: klass
            for name, klass in cls._registry.items()
            if getattr(klass, "meta", None) and getattr(klass.meta, "category", None) == category
        }
