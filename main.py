"""
main.py
-------
FastAPI Backend. Orchestrates ingestion and RAG querying.
"""

import os
import logging
import sys
import shutil
import tempfile
import urllib.request
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List

from config import settings
from engine import InsuranceRAGEngine
from rag_ingestion.pipeline import ExtractionPipeline
from schemas import (
    SingleQueryRequest,
    CompareQueryRequest,
    UrlIngestRequest,
    CollectionListResponse,
    StandardResponse,
)

#  Set up a basic console handler so logs actually have a place to print
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Insurance RAG API")

# Initialize the Engine using central settings
rag_engine = InsuranceRAGEngine(gemini_api_key=settings.gemini_api_key)

# --- QUERY ENDPOINTS ---


@app.post("/api/v1/query/single", response_model=StandardResponse)
async def query_single(req: SingleQueryRequest):
    try:
        answer = await rag_engine.a_query_single_policy(
            query=req.query,
            collection_name=req.collection_name,
            history=req.history,
            retrieve_top_k=req.retrieve_top_k,
            rerank_top_k=req.rerank_top_k,
        )
        return StandardResponse(
            status="success",
            message="Query processed successfully.",
            data={"markdown_report": answer},
        )
    except Exception as e:
        logger.error(f"Error processing single query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/compare", response_model=StandardResponse)
async def query_compare(req: CompareQueryRequest):
    try:
        answer = await rag_engine.a_compare_policies(
            query=req.query,
            collection_a=req.collection_a,
            collection_b=req.collection_b,
            history=req.history,
            retrieve_top_k=req.retrieve_top_k,
            rerank_top_k=req.rerank_top_k,
        )
        return StandardResponse(
            status="success",
            message="Comparison processed successfully.",
            data={"markdown_report": answer},
        )
    except Exception as e:
        logger.error(f"Error processing comparison query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- ADMIN ENDPOINTS ---


@app.get("/api/v1/admin/collections", response_model=CollectionListResponse)
async def list_collections():
    try:
        if settings.vector_db_type == "pinecone":
            import pinecone

            pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index_name)
            # Pinecone namespaces act as our collections
            stats = index.describe_index_stats()
            collections = list(stats["namespaces"].keys())
        else:
            import chromadb

            client = chromadb.PersistentClient(path=settings.chroma_dir)
            collections = [c.name for c in client.list_collections()]

        return CollectionListResponse(collections=collections)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch collections: {str(e)}"
        )


@app.delete("/api/v1/admin/collections/{collection_name}")
async def delete_collection(collection_name: str):
    try:
        if settings.vector_db_type == "pinecone":
            import pinecone

            pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
            idx = pc.Index(settings.pinecone_index_name)
            idx.delete(delete_all=True, namespace=collection_name)
        else:
            import chromadb

            client = chromadb.PersistentClient(path=settings.chroma_dir)
            client.delete_collection(collection_name)

        # Also clean up the local BM25 pickle from the new dedicated directory
        bm25_path = os.path.join(settings.bm25_dir, f"{collection_name}_bm25.pkl")
        if os.path.exists(bm25_path):
            os.remove(bm25_path)  # Uses os.remove because it is a file, not a directory

        return StandardResponse(
            status="success", message=f"Collection {collection_name} deleted"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/admin/ingest/url")
async def ingest_url(req: UrlIngestRequest):
    """
    Downloads and indexes a PDF from a provided URL using non-blocking network I/O.
    """
    tmp_path = ""
    try:
        # 1. Execute non-blocking network request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    str(req.url),
                    headers={"User-Agent": "Mozilla/5.0"},
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()
            except httpx.RequestError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Error requesting {exc.request.url}."
                )
            except httpx.HTTPStatusError as exc:
                raise HTTPException(
                    status_code=exc.response.status_code,
                    detail=f"Error response {exc.response.status_code} while requesting {exc.request.url}.",
                )

        # 2. Write the retrieved bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # 3. Execute the asynchronous extraction pipeline
        await ExtractionPipeline(
            pdf_path=tmp_path, collection_name=req.collection_name
        ).a_run()

        return StandardResponse(status="success", message="URL indexed successfully.")

    finally:
        # 4. Ensure cleanup occurs regardless of pipeline success/failure
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/v1/admin/ingest/url")
async def ingest_url(req: UrlIngestRequest):
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = urllib.request.urlopen(
                urllib.request.Request(str(req.url), headers=headers)
            )
            tmp.write(r.read())
            tmp_path = tmp.name
        await ExtractionPipeline(
            pdf_path=tmp_path, collection_name=req.collection_name
        ).a_run()
        return StandardResponse(status="success", message="URL indexed")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
