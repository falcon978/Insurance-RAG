"""
main.py
-------
FastAPI application entry point. Configures lifespan, models, middleware, and routers.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from rag.factories import InferenceModelFactory
from rag.rerankers import ContextReranker
from rag.generator import ResponseGenerator
from rag.query_rewriter import get_structured_rewriter_chain
from api.exception_handlers import register_exception_handlers
from api.routers.query import router as query_router
from api.routers.admin import router as admin_router

# Set up standard console logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# --- CONNECTION POOL & MODEL LIFECYCLE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes shared database connection pools and pre-warms ML models.
    Stores instances on application state for dependency injection.
    """
    import redis.asyncio as redis

    logger.info("Initializing Database Connection Pools...")

    # 1. Initialize Redis (TCP Pool)
    app.state.redis = redis.from_url(
        settings.redis_url,
        decode_responses=True,
        health_check_interval=30,
        retry_on_timeout=True,
        max_connections=settings.redis_max_connections,
    )

    # 2. Initialize Pinecone (HTTP Session Pool)
    if settings.vector_db_type == "pinecone":
        import pinecone

        pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
        app.state.pinecone_index = pc.Index(settings.pinecone_index_name)
    else:
        app.state.pinecone_index = None

    # 3. Initialize Chroma (Persistent Disk Client)
    if settings.vector_db_type == "chroma":
        import chromadb

        app.state.chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
    else:
        app.state.chroma_client = None

    # 4. Pre-warm ML Models
    logger.info("Initializing ML Models (Embeddings, Reranker, LLMs)...")
    app.state.embeddings = HuggingFaceEmbeddings(
        model_name=settings.embed_model_name,
        model_kwargs={"device": settings.hf_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    app.state.reranker = ContextReranker(
        model_name=settings.rerank_model_name, device=settings.hf_device
    )
    generator_llm = InferenceModelFactory.get_chat_model(
        model_name=settings.llm_model_name, temperature=0.0
    )
    app.state.generator = ResponseGenerator(llm=generator_llm)

    planner_llm = InferenceModelFactory.get_chat_model(
        model_name=settings.planner_model_name,
        temperature=settings.planner_temperature,
    )
    app.state.rewriter_chain = get_structured_rewriter_chain(planner_llm)

    yield  # Application serves requests here

    logger.info("Closing database connections...")
    await app.state.redis.aclose()


# --- APPLICATION INITIALIZATION ---
app = FastAPI(title="Insurance Policy RAG API", lifespan=lifespan)

# Register custom domain exception handlers
register_exception_handlers(app)

# Register API Routers
app.include_router(query_router)
app.include_router(admin_router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirects the root URL to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")
