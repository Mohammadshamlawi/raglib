"""
Example plugin demonstrating how to create and register a custom RAG technique.

This plugin shows how to:
1. Create a custom RAGTechnique subclass
2. Register it with the TechniqueRegistry
3. Make it discoverable via the plugin system
"""

from raglib.core import RAGTechnique, TechniqueMeta
from raglib.registry import TechniqueRegistry
from raglib.schemas import Document, RagResult
from typing import List


@TechniqueRegistry.register("example_keyword_filter")
class ExampleKeywordFilterTechnique(RAGTechnique):
    """
    Example technique that filters documents based on keyword presence.
    
    This is a simple demonstration technique that shows how to:
    - Implement the RAGTechnique interface
    - Use TechniqueMeta for metadata
    - Return proper RagResult objects
    """
    
    meta = TechniqueMeta(
        name="Example Keyword Filter",
        description="Filters documents containing specific keywords and returns them ranked by keyword frequency",
        author="RAGLib Example Team",
        email="examples@raglib.org", 
        version="1.0.0",
        category="retrieval",
        tags=["example", "keyword", "filter", "demo"],
        requirements=[]  # No external dependencies
    )
    
    def __init__(self, keywords: List[str] = None, case_sensitive: bool = False, **kwargs):
        """
        Initialize the keyword filter technique.
        
        Args:
            keywords: List of keywords to filter by. If None, uses the query.
            case_sensitive: Whether keyword matching should be case-sensitive
        """
        super().__init__(**kwargs)
        self.keywords = keywords or []
        self.case_sensitive = case_sensitive
    
    def apply(
        self, 
        query: str, 
        corpus: List[Document], 
        top_k: int = 5, 
        **kwargs
    ) -> RagResult:
        """
        Apply keyword filtering to the corpus.
        
        Args:
            query: The search query (used as keywords if none specified)
            corpus: List of documents to filter
            top_k: Maximum number of documents to return
            **kwargs: Additional parameters
            
        Returns:
            RagResult with filtered and ranked documents
        """
        # Use query words as keywords if none specified
        search_keywords = self.keywords if self.keywords else query.split()
        
        # Score documents based on keyword frequency
        scored_docs = []
        
        for doc in corpus:
            score = self._calculate_keyword_score(doc.text, search_keywords)
            if score > 0:  # Only include documents with matching keywords
                scored_docs.append((doc, score))
        
        # Sort by score (descending) and take top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:top_k]]
        
        return RagResult(
            query=query,
            retrieved_documents=top_docs,
            metadata={
                "technique": "example_keyword_filter",
                "keywords_used": search_keywords,
                "case_sensitive": self.case_sensitive,
                "total_matches": len(scored_docs),
                "scores": [score for _, score in scored_docs[:top_k]],
                **kwargs
            }
        )
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate keyword score for a document.
        
        Args:
            text: Document text to score
            keywords: List of keywords to search for
            
        Returns:
            Score based on keyword frequency (higher = more relevant)
        """
        if not self.case_sensitive:
            text = text.lower()
            keywords = [k.lower() for k in keywords]
        
        score = 0.0
        words = text.split()
        
        for keyword in keywords:
            # Count exact matches
            keyword_count = words.count(keyword)
            score += keyword_count
            
            # Bonus for partial matches (substring)
            partial_matches = sum(1 for word in words if keyword in word and word != keyword)
            score += partial_matches * 0.5
        
        return score


# This technique will be registered when the plugin is loaded
# You can also manually register additional techniques:

@TechniqueRegistry.register("example_length_filter") 
class ExampleLengthFilterTechnique(RAGTechnique):
    """Example technique that filters documents by text length."""
    
    meta = TechniqueMeta(
        name="Example Length Filter",
        description="Filters documents based on text length constraints",
        author="RAGLib Example Team", 
        email="examples@raglib.org",
        version="1.0.0",
        category="preprocessing",
        tags=["example", "length", "filter"],
        requirements=[]
    )
    
    def __init__(self, min_length: int = 10, max_length: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
    
    def apply(
        self, 
        query: str, 
        corpus: List[Document], 
        top_k: int = 5,
        **kwargs
    ) -> RagResult:
        """Filter documents by length and return them."""
        filtered_docs = []
        
        for doc in corpus:
            text_length = len(doc.text)
            if self.min_length <= text_length <= self.max_length:
                filtered_docs.append(doc)
        
        # Return top_k documents (no special ranking)
        result_docs = filtered_docs[:top_k]
        
        return RagResult(
            query=query,
            retrieved_documents=result_docs,
            metadata={
                "technique": "example_length_filter", 
                "min_length": self.min_length,
                "max_length": self.max_length,
                "filtered_count": len(filtered_docs),
                **kwargs
            }
        )


# Plugin information for discovery
__plugin_name__ = "raglib_example_plugin"
__plugin_version__ = "1.0.0" 
__plugin_description__ = "Example plugin showing how to create custom RAG techniques"
