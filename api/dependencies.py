"""
api/dependencies.py
-------------------
FastAPI dependency injection providers for storage clients and the RAG engine.
"""

from fastapi import Depends, Request

from config import settings
from engine import InsuranceRAGEngine


async def get_redis(request: Request):
    """Retrieves the active Redis connection pool from application state."""
    return getattr(request.app.state, "redis", None)


async def get_vector_client(request: Request):
    """Retrieves the configured vector database client from application state."""
    if settings.vector_db_type == "pinecone":
        return getattr(request.app.state, "pinecone_index", None)
    elif settings.vector_db_type == "chroma":
        return getattr(request.app.state, "chroma_client", None)
    return None


async def get_lexical_client(request: Request):
    """Retrieves the configured lexical database client from application state."""
    if settings.lexical_db_type == "redis":
        return getattr(request.app.state, "redis", None)
    return None


async def get_rag_engine(
    request: Request,
    vector_client=Depends(get_vector_client),
    lexical_client=Depends(get_lexical_client),
) -> InsuranceRAGEngine:
    """Instantiates and returns an InsuranceRAGEngine injected with pre-warmed models and clients."""
    return InsuranceRAGEngine(
        embeddings=request.app.state.embeddings,
        reranker=request.app.state.reranker,
        generator=request.app.state.generator,
        rewriter_chain=request.app.state.rewriter_chain,
        vector_client=vector_client,
        lexical_client=lexical_client,
    )
