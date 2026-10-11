"""
config.py
---------
Centralized configuration management using Pydantic Settings.
Toggles between 'chroma'/'pinecone' for dense vector storage and
'ram'/'redis' for lexical BM25 search caching/execution.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Literal, Optional


class Settings(BaseSettings):
    @field_validator("vector_db_type", "lexical_db_type", "inference_provider", mode="before")
    @classmethod
    def force_lower_case(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v
    # Toggle between 'chroma' (local) and 'pinecone' (cloud)
    vector_db_type: Literal["chroma", "pinecone"] = os.getenv(  # type: ignore[assignment]
        "VECTOR_DB_TYPE", "chroma"
    )

    # Chroma Config (used for Vector DB if type=chroma, AND for local BM25 cache)
    chroma_dir: str = "./chroma_data"

    # Pinecone Config
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None

    # Lexical DB Config
    lexical_db_type: Literal["ram", "redis"] = "ram"
    bm25_dir: str = "./bm25_cache"
    redis_url: Optional[str] = None
    redis_max_connections: int = 50

    # MODEL CONFIGURATION
    embed_model_name: str = "BAAI/bge-large-en-v1.5"
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    llm_model_name: str = "gemini-3-flash-preview"

    # Planner LLM Configs
    planner_model_name: str = "gemini-3.1-flash-lite"
    planner_temperature: float = 0.0

    # Default Retrieval Hyperparameters
    default_retrieve_top_k: int = 15
    default_rerank_top_k: int = 5
    default_semantic_weight: float = 0.5
    default_lexical_weight: float = 0.5

    # Default Ingestion Hyperparameters
    default_chunk_size: int = 2000
    default_chunk_overlap: int = 300

    # Frontend / Conversational Settings
    max_history_messages: int = 6

    # LLM and Device Config
    inference_provider: str = "gemini"  # "gemini" or "groq"
    admin_password: str = "admin123"
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    hf_device: str = "cpu"  # Set to 'cuda' for GPU acceleration

    # LangSmith Config
    langsmith_tracing_v2: str = "false"
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


# Force inject telemetry vars into os.environ for LangChain's internal hooks
if settings.langsmith_tracing_v2.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
