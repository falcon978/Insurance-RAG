"""
rag/factories.py
----------------
Implements the Factory pattern to abstract database initialization.
Ensures the core RAG engine remains completely agnostic to the underlying infrastructure.
"""

import os
import pickle
import logging
from typing import Optional

from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings

try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

from config import settings
from rag.retriever import UpstashRediSearchRetriever

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory for initializing Dense Vector Databases (Semantic Search)."""

    @staticmethod
    def get_vector_store(collection_name: str, embeddings: Embeddings) -> VectorStore:
        if settings.vector_db_type == "pinecone":
            if not PineconeVectorStore or not settings.pinecone_api_key:
                raise ValueError("Pinecone is not installed or the API key is missing.")

            logger.info(f"Connecting to Pinecone namespace: {collection_name}")
            return PineconeVectorStore(
                index_name=settings.pinecone_index_name,
                embedding=embeddings,
                namespace=collection_name,
                pinecone_api_key=settings.pinecone_api_key,
            )

        elif settings.vector_db_type == "chroma":
            logger.info(f"Connecting to Chroma local collection: {collection_name}")
            return Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=settings.chroma_dir,
            )

        else:
            raise ValueError(f"Unsupported vector_db_type: {settings.vector_db_type}")


class LexicalStoreFactory:
    """Factory for initializing Lexical Databases (Keyword/BM25 Search)."""

    @staticmethod
    def get_lexical_retriever(collection_name: str, top_k: int):
        if settings.lexical_db_type == "redis":
            logger.info(f"Connecting to Upstash RediSearch index: {collection_name}")
            return UpstashRediSearchRetriever(
                redis_url=settings.redis_url,
                collection_name=collection_name,
                k=top_k,
            )

        elif settings.lexical_db_type == "ram":
            return LexicalStoreFactory._load_ram_bm25(collection_name, top_k)

        else:
            raise ValueError(f"Unsupported lexical_db_type: {settings.lexical_db_type}")

    @staticmethod
    def _load_ram_bm25(collection_name: str, top_k: int) -> Optional[BM25Retriever]:
        bm25_path = os.path.join(settings.bm25_dir, f"{collection_name}_bm25.pkl")
        if not os.path.exists(bm25_path):
            logger.warning(
                f"No RAM BM25 index found at {bm25_path}. Lexical track will be bypassed."
            )
            return None

        try:
            with open(bm25_path, "rb") as f:
                documents = pickle.load(f)
            if not documents:
                return None

            retriever = BM25Retriever.from_documents(documents)
            retriever.k = top_k
            logger.info(f"Loaded RAM BM25 index for: {collection_name}")
            return retriever
        except Exception as e:
            logger.error(f"Failed to load RAM BM25 index: {e}")
            return None
