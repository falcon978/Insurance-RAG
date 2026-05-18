"""
main.py
-------
FastAPI Backend. Orchestrates ingestion and RAG querying.
"""

import os

# --- HARDCODE THREAD LIMITS TO PREVENT CPU THRASHING ---
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevents HF tokenizer thread warnings

import logging
import sys
import shutil
import tempfile
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.responses import RedirectResponse

from config import settings
from engine import InsuranceRAGEngine
from rag_ingestion.pipeline import ExtractionPipeline
from rag_ingestion.indexer import PolicyVectorStore
from schemas import (
    SingleQueryRequest,
    CompareQueryRequest,
    UrlIngestRequest,
    CollectionListResponse,
    StandardResponse,
)

# Set up a basic console handler for observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# --- CONNECTION POOL LIFECYCLE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    import redis.asyncio as redis

    logger.info("Initializing Database Connection Pools...")

    # 1. Initialize Redis (TCP Pool)
    app.state.redis = redis.from_url(
        settings.redis_url,
        decode_responses=True,
        health_check_interval=30,
        retry_on_timeout=True,
    )

    # 2. Initialize Pinecone (HTTP Session Pool)
    if settings.vector_db_type == "pinecone":
        import pinecone

        pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
        app.state.pinecone_index = pc.Index(settings.pinecone_index_name)
    else:
        app.state.pinecone_index = None

    yield  # App processes requests here

    logger.info("Closing database connections...")
    await app.state.redis.aclose()


app = FastAPI(title="Insurance RAG API", lifespan=lifespan)


# --- DEPENDENCY INJECTION ---
async def get_redis(request: Request):
    return request.app.state.redis


async def get_pinecone_index(request: Request):
    return request.app.state.pinecone_index


# Inject connections into the RAG Engine dynamically per-request
async def get_rag_engine(
    redis_client=Depends(get_redis), pinecone_index=Depends(get_pinecone_index)
):
    return InsuranceRAGEngine(
        gemini_api_key=settings.gemini_api_key,
        redis_client=redis_client,
        pinecone_index=pinecone_index,
    )


@app.get("/", include_in_schema=False)
async def root():
    """Redirects the root URL to the interactive Swagger UI."""
    return RedirectResponse(url="/docs")


# --- HELPER FUNCTION ---
async def check_if_already_indexed(
    collection_name: str, redis_client, pinecone_index
) -> bool:
    """
    Instantiates the VectorStore to run a lightning-fast, zero-cost
    database check before wasting CPU on PDF parsing.
    """
    store = PolicyVectorStore(
        collection_name=collection_name,
        redis_client=redis_client,
        pinecone_index=pinecone_index,
    )
    return await store._a_is_namespace_populated()


# --- QUERY ENDPOINTS ---


@app.post("/api/v1/query/single", response_model=StandardResponse)
async def query_single(
    req: SingleQueryRequest, rag_engine: InsuranceRAGEngine = Depends(get_rag_engine)
):
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
            message="Query processed",
            data={"markdown_report": answer},
        )
    except Exception as e:
        logger.error(f"Error processing single query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/compare", response_model=StandardResponse)
async def query_compare(
    req: CompareQueryRequest, rag_engine: InsuranceRAGEngine = Depends(get_rag_engine)
):
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
            message="Comparison processed",
            data={"markdown_report": answer},
        )
    except Exception as e:
        logger.error(f"Error processing comparison query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- ADMIN ENDPOINTS ---


@app.post("/api/v1/admin/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    chunk_size: int = Form(settings.default_chunk_size),
    chunk_overlap: int = Form(settings.default_chunk_overlap),
    redis_client=Depends(get_redis),
    pinecone_index=Depends(get_pinecone_index),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # EARLY EXIT: Prevent wasteful disk write and CPU chunking
    if await check_if_already_indexed(collection_name, redis_client, pinecone_index):
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
            redis_client=redis_client,
            pinecone_index=pinecone_index,
        ).a_run()

        return StandardResponse(
            status="success", message=f"Successfully indexed {file.filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/v1/admin/ingest/url")
async def ingest_url(
    req: UrlIngestRequest,
    redis_client=Depends(get_redis),
    pinecone_index=Depends(get_pinecone_index),
):
    """Downloads and indexes a PDF from a provided URL using non-blocking network I/O."""

    # EARLY EXIT: Prevent wasteful network download and CPU chunking
    if await check_if_already_indexed(
        req.collection_name, redis_client, pinecone_index
    ):
        return StandardResponse(
            status="success",
            message=f"Skipped: Collection '{req.collection_name}' is already indexed.",
        )

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
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Error requesting URL: {exc}"
                )

        # 2. Write the retrieved bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # 3. Execute the asynchronous extraction pipeline
        await ExtractionPipeline(
            pdf_path=tmp_path,
            collection_name=req.collection_name,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            redis_client=redis_client,
            pinecone_index=pinecone_index,
        ).a_run()

        return StandardResponse(status="success", message="URL indexed successfully.")
    finally:
        # 4. Ensure cleanup occurs regardless of pipeline success/failure
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/api/v1/admin/collections", response_model=CollectionListResponse)
def list_collections(pinecone_index=Depends(get_pinecone_index)):
    try:
        collections = []
        if settings.vector_db_type == "pinecone" and pinecone_index is not None:
            stats = pinecone_index.describe_index_stats()
            collections = list(stats.namespaces.keys())
        else:
            import chromadb

            client = chromadb.PersistentClient(path=settings.chroma_dir)
            collections = [col.name for col in client.list_collections()]
        return CollectionListResponse(collections=collections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/admin/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    redis_client=Depends(get_redis),
    pinecone_index=Depends(get_pinecone_index),
):
    try:
        # 1. Drop Vector DB Namespace
        if settings.vector_db_type == "pinecone" and pinecone_index:
            pinecone_index.delete(delete_all=True, namespace=collection_name)
        else:
            import chromadb

            client = chromadb.PersistentClient(path=settings.chroma_dir)
            client.delete_collection(collection_name)

        # 2. Drop Lexical DB Index
        if settings.lexical_db_type == "redis" and redis_client:
            try:
                # 'DD' drops the index AND deletes all underlying hash documents
                await redis_client.ft(f"idx:{collection_name}").dropindex(
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
