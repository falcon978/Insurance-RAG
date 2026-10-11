"""
rag/exceptions.py
-----------------
Domain-specific exceptions for the Insurance RAG Engine.
"""

class RAGCoreException(Exception):
    """Base exception for all Insurance RAG errors."""
    pass

class VectorStoreError(RAGCoreException):
    """Raised when the vector database (Pinecone/Chroma) fails."""
    pass

class LexicalStoreError(RAGCoreException):
    """Raised when the lexical database (Redis/RAM) fails."""
    pass

class DocumentParsingError(RAGCoreException):
    """Raised when PDF extraction or chunking fails."""
    pass

class LLMGenerationError(RAGCoreException):
    """Raised when the LLM service (Gemini) times out or fails to parse structured output."""
    pass

class CollectionNotFoundError(RAGCoreException):
    """Raised when querying a collection that does not exist."""
    pass
