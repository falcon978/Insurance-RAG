"""
api/exception_handlers.py
-------------------------
Registers custom exception handlers mapping domain exceptions to HTTP responses.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag.exceptions import (
    VectorStoreError,
    LexicalStoreError,
    DocumentParsingError,
    LLMGenerationError,
    CollectionNotFoundError,
)


def register_exception_handlers(app: FastAPI) -> None:
    """Registers domain exception handlers on the provided FastAPI application instance."""

    @app.exception_handler(VectorStoreError)
    async def vector_store_exception_handler(request: Request, exc: VectorStoreError):
        return JSONResponse(
            status_code=503,
            content={"detail": f"Vector Database Error: {str(exc)}"},
        )

    @app.exception_handler(LexicalStoreError)
    async def lexical_store_exception_handler(request: Request, exc: LexicalStoreError):
        return JSONResponse(
            status_code=503,
            content={"detail": f"Lexical Database Error: {str(exc)}"},
        )

    @app.exception_handler(DocumentParsingError)
    async def document_parsing_exception_handler(
        request: Request, exc: DocumentParsingError
    ):
        return JSONResponse(
            status_code=400,
            content={"detail": f"Document Processing Error: {str(exc)}"},
        )

    @app.exception_handler(LLMGenerationError)
    async def llm_generation_exception_handler(request: Request, exc: LLMGenerationError):
        return JSONResponse(
            status_code=502,
            content={"detail": f"AI Generation Error: {str(exc)}"},
        )

    @app.exception_handler(CollectionNotFoundError)
    async def collection_not_found_exception_handler(
        request: Request, exc: CollectionNotFoundError
    ):
        return JSONResponse(
            status_code=404,
            content={"detail": f"Collection Not Found: {str(exc)}"},
        )
