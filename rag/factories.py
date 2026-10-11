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
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

from config import settings
from rag.retriever import RedisSearchRetriever
from rag.exceptions import VectorStoreError, LexicalStoreError

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory for initializing Dense Vector Databases (Semantic Search)."""

    @staticmethod
    def get_vector_store(
        collection_name: str,
        embeddings: Embeddings,
        vector_client=None,
    ) -> VectorStore:
        if settings.vector_db_type == "pinecone":
            if not PineconeVectorStore or vector_client is None:
                raise VectorStoreError("Pinecone is not installed or the API key is missing.")

            logger.info(f"Connecting to Pinecone namespace: {collection_name}")
            return PineconeVectorStore(
                index=vector_client, embedding=embeddings, namespace=collection_name
            )

        elif settings.vector_db_type == "chroma":
            logger.info(f"Connecting to Chroma local collection: {collection_name}")
            if vector_client is not None:
                return Chroma(
                    client=vector_client,
                    collection_name=collection_name,
                    embedding_function=embeddings,
                )
            return Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=settings.chroma_dir,
            )

        else:
            raise VectorStoreError(f"Unsupported vector_db_type: {settings.vector_db_type}")


class LexicalStoreFactory:
    """Factory for initializing Lexical Databases (Keyword/BM25 Search)."""

    @staticmethod
    def get_lexical_retriever(
        collection_name: str,
        top_k: int,
        lexical_client=None,
    ):
        if settings.lexical_db_type == "redis":
            logger.info(f"Connecting to RediSearch index: {collection_name}")
            return RedisSearchRetriever(
                redis_client=lexical_client,
                collection_name=collection_name,
                k=top_k,
            )

        elif settings.lexical_db_type == "ram":
            return LexicalStoreFactory._load_ram_bm25(collection_name, top_k)

        else:
            raise LexicalStoreError(f"Unsupported lexical_db_type: {settings.lexical_db_type}")

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
            raise LexicalStoreError(f"Failed to load RAM BM25 index: {e}")


class InferenceModelFactory:
    """Factory for initializing the primary RAG Chat Model."""

    @staticmethod
    def get_chat_model(model_name: str, temperature: float = 0.0) -> BaseChatModel:
        if settings.inference_provider == "gemini":
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is missing.")
            logger.info(f"Connecting to Gemini Inference Model: {model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                api_key=settings.gemini_api_key,
            )
        elif settings.inference_provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is missing.")
            logger.info(f"Connecting to Groq Inference Model: {model_name}")
            return ChatGroq(
                model=model_name,
                temperature=temperature,
                groq_api_key=settings.groq_api_key,
            )
        else:
            raise ValueError(f"Unsupported inference provider: {settings.inference_provider}")
