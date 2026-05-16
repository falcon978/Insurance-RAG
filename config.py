"""
config.py
---------
Centralized configuration management using Pydantic Settings.
Toggles between 'chroma'/'pinecone' for dense vector storage and
'ram'/'upstash' for lexical BM25 search caching/execution.
"""

import os
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # Toggle between 'chroma' (local) and 'pinecone' (cloud)
    vector_db_type: Literal["chroma", "pinecone"] = os.getenv(
        "VECTOR_DB_TYPE", "chroma"
    )

    # Chroma Config (used for Vector DB if type=chroma, AND for local BM25 cache)
    chroma_dir: str = "./chroma_data"

    # Pinecone Config
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME")

    # Lexical DB Config
    lexical_db_type: Literal["ram", "upstash"] = os.getenv("LEXICAL_DB_TYPE", "ram")
    bm25_dir: str = "./bm25_cache"
    upstash_redis_url: str = os.getenv("UPSTASH_REDIS_URL")

    # MODEL CONFIGURATION
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    rerank_model_name: str = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gemini-3-flash-preview")

    # Planner LLM Configs
    planner_model_name: str = os.getenv(
        "PLANNER_LLM_MODEL_NAME", "gemini-3.1-flash-lite"
    )
    planner_temperature: float = 0.0

    # Default Retrieval Hyperparameters
    default_retrieve_top_k: int = int(os.getenv("DEFAULT_RETRIEVE_TOP_K", 15))
    default_rerank_top_k: int = int(os.getenv("DEFAULT_RERANK_TOP_K", 5))
    default_semantic_weight: float = float(os.getenv("DEFAULT_SEMANTIC_WEIGHT", 0.5))
    default_lexical_weight: float = float(os.getenv("DEFAULT_LEXICAL_WEIGHT", 0.5))

    # Default Ingestion Hyperparameters
    default_chunk_size: int = int(os.getenv("CHUNK_SIZE", 2200))
    default_chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 400))

    # LLM and Device Config
    admin_password: str = os.getenv("ADMIN_PASSWORD", "admin123")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY")
    hf_device: str = os.getenv("HF_DEVICE", "cpu")  # Set to 'cuda' for GPU acceleration

    # LangSmith Config
    langsmith_tracing_v2: str = os.getenv("LANGSMITH_TRACING_V2", "false")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


# Force inject telemetry vars into os.environ for LangChain's internal hooks
if settings.langsmith_tracing_v2.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
