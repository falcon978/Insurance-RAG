"""
api/routers/admin.py
-------------------
Defines administrative endpoints for document ingestion and collection lifecycle management.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import httpx

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from config import settings
from rag_ingestion.pipeline import ExtractionPipeline
from rag_ingestion.indexer import PolicyVectorStore
from api.dependencies import get_vector_client, get_lexical_client
from schemas import (
    UrlIngestRequest,
    CollectionListResponse,
    StandardResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


async def check_if_already_indexed(
    collection_name: str, vector_client=None, lexical_client=None
) -> bool:
    """
    Instantiates the PolicyVectorStore to execute a fast namespace population check
    before proceeding with computationally expensive PDF extraction and chunking.
    """
    store = PolicyVectorStore(
        collection_name=collection_name,
        vector_client=vector_client,
        lexical_client=lexical_client,
    )
    return await store._a_is_namespace_populated()


@router.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    chunk_size: int = Form(settings.default_chunk_size),
    chunk_overlap: int = Form(settings.default_chunk_overlap),
    vector_client=Depends(get_vector_client),
    lexical_client=Depends(get_lexical_client),
):
    """Handles multipart PDF upload and executes the extraction and indexing pipeline."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if await check_if_already_indexed(collection_name, vector_client, lexical_client):
        return StandardResponse(
            status="success",
            message=f"Skipped: Collection '{collection_name}' is already indexed.",
        )

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        await ExtractionPipeline(
            pdf_path=tmp_path,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vector_client=vector_client,
            lexical_client=lexical_client,
        ).a_run()

        return StandardResponse(
            status="success", message=f"Successfully indexed {file.filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/ingest/url")
async def ingest_url(
    req: UrlIngestRequest,
    vector_client=Depends(get_vector_client),
    lexical_client=Depends(get_lexical_client),
):
    """Downloads a PDF from a remote URL via asynchronous HTTP and triggers extraction."""
    if await check_if_already_indexed(
        req.collection_name, vector_client, lexical_client
    ):
        return StandardResponse(
            status="success",
            message=f"Skipped: Collection '{req.collection_name}' is already indexed.",
        )

    tmp_path = ""
    try:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    str(req.url),
                    headers={"User-Agent": "Mozilla/5.0"},
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Error requesting URL: {exc}"
                )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        await ExtractionPipeline(
            pdf_path=tmp_path,
            collection_name=req.collection_name,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            vector_client=vector_client,
            lexical_client=lexical_client,
        ).a_run()

        return StandardResponse(status="success", message="URL indexed successfully.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/collections", response_model=CollectionListResponse)
def list_collections(
    vector_client=Depends(get_vector_client),
):
    """Lists all available policy namespaces/collections across the configured vector database."""
    try:
        collections = []
        if settings.vector_db_type == "pinecone" and vector_client is not None:
            stats = vector_client.describe_index_stats()
            collections = list(stats.namespaces.keys())
        elif settings.vector_db_type == "chroma" and vector_client is not None:
            collections = [col.name for col in vector_client.list_collections()]
        return CollectionListResponse(collections=collections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    vector_client=Depends(get_vector_client),
    lexical_client=Depends(get_lexical_client),
):
    """Drops the specified collection from both vector and lexical persistent stores."""
    try:
        # 1. Drop Vector DB Namespace
        if settings.vector_db_type == "pinecone" and vector_client:
            await asyncio.to_thread(
                vector_client.delete, delete_all=True, namespace=collection_name
            )
        elif settings.vector_db_type == "chroma" and vector_client:
            await asyncio.to_thread(vector_client.delete_collection, collection_name)

        # 2. Drop Lexical DB Index
        if settings.lexical_db_type == "redis" and lexical_client:
            try:
                await lexical_client.ft(f"idx:{collection_name}").dropindex(
                    delete_documents=True
                )
            except Exception as e:
                logger.warning(f"Failed to drop RediSearch index (may not exist): {e}")
        else:
            bm25_path = os.path.join(settings.bm25_dir, f"{collection_name}_bm25.pkl")
            if os.path.exists(bm25_path):
                os.remove(bm25_path)

        return StandardResponse(
            status="success", message=f"Collection {collection_name} deleted"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
