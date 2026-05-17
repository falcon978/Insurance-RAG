"""
indexer.py
-----------------------
Phase 4: Vector Embedding and Lexical Indexing.

This module acts as the data ingestion DAO (Data Access Object). It delegates
the instantiation of the Vector DB to the central Factory, while retaining
the specialized pipeline logic required to batch upload documents to
local RAM pickles or a serverless Upstash RediSearch engine.

Key Features:
  - Idempotent Upserts: Uses a custom `chunk_id` so re-running the script
    updates existing chunks rather than duplicating them.
  - Namespace Checks: Checks if the vector namespace is already populated before hitting APIs.
  - Concurrent Safety: Implements global async locks for file I/O operations.
"""

import asyncio
import os
import pickle
import logging
from typing import List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Internal Module Imports
from config import settings
from rag.factories import VectorStoreFactory
from .models import PolicyChunk

logger = logging.getLogger(__name__)


class PolicyVectorStore:

    # Class-level attribute to ensure a globally shared lock across all concurrent instances.
    _bm25_io_lock = None

    def __init__(
        self,
        collection_name: str,
        device: str = "cpu",
        redis_client=None,
        pinecone_index=None,
    ):
        """
        Initializes the embedding model and delegates Vector DB connection to the Factory.
        """
        # Lazy instantiation ensures the lock is safely bound to the active asyncio event loop.
        if PolicyVectorStore._bm25_io_lock is None:
            PolicyVectorStore._bm25_io_lock = asyncio.Lock()

        self.collection_name = collection_name
        self.device = device

        # Store clients for ingestion
        self.redis_client = redis_client
        self.pinecone_index = pinecone_index

        # 1. Initialize Shared Embedding Model Dynamically
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},  # Enforces L2 Normalization
        )

        # 2. Delegate Vector Database creation to the Factory
        self.vector_store = VectorStoreFactory.get_vector_store(
            collection_name=self.collection_name,
            embeddings=self.embeddings,
            pinecone_index=self.pinecone_index,
        )

    def _prepare_documents(
        self, chunks: List[PolicyChunk]
    ) -> Tuple[List[Document], List[str]]:
        """Transforms raw PolicyChunks into Langchain Documents with flattened metadata."""
        documents = []
        ids = []

        for chunk in chunks:
            # Flatten metadata for database cross-compatibility
            # Note: We inject chunk_id into metadata here to help with BM25 deduplication later
            flat_metadata = {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "sub_section": chunk.sub_section,
                "heading": chunk.heading,
                "token_estimate": chunk.token_estimate,
                "has_table": chunk.metadata.get("has_table", False),
                "has_list": chunk.metadata.get("has_list", False),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "char_count": chunk.metadata.get("char_count", 0),
            }

            doc = Document(page_content=chunk.text, metadata=flat_metadata)
            documents.append(doc)
            ids.append(chunk.chunk_id)

        return documents, ids

    async def _a_is_namespace_populated(self) -> bool:
        """
        Non-blocking administrative check to verify if the collection contains existing vectors.
        This prevents redundant embedding calculations and API calls.
        """

        def check():
            if settings.vector_db_type == "pinecone":
                try:
                    # Fast check using the existing internal Pinecone client
                    stats = self.vector_store._index.describe_index_stats()

                    # Handle Pinecone SDK variations (Object vs Dict)
                    namespaces = (
                        stats.namespaces
                        if hasattr(stats, "namespaces")
                        else stats.get("namespaces", {})
                    )

                    namespace_data = namespaces.get(self.collection_name, {})

                    # Safely extract vector_count whether namespace_data is a dict or an object
                    vector_count = (
                        namespace_data.get("vector_count", 0)
                        if isinstance(namespace_data, dict)
                        else getattr(namespace_data, "vector_count", 0)
                    )

                    return vector_count > 0
                except Exception:
                    return False
            else:
                try:
                    # Chroma-specific ID extraction
                    return self.vector_store._collection.count() > 0
                except Exception:
                    return False

        return await asyncio.to_thread(check)

    async def _a_upsert_to_vector_db(
        self, documents: List[Document], ids: List[str], batch_size: int
    ):
        """Asynchronous batch upsert to the configured Vector DB."""
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logging.info(
            f"Adding {len(documents)} chunks to {settings.vector_db_type.upper()} in {total_batches} batches..."
        )

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            # Use LangChain's native async add method (safely threads if provider lacks native async)
            await self.vector_store.aadd_documents(documents=batch_docs, ids=batch_ids)
            logging.info(f"  Processed batch {(i//batch_size) + 1}/{total_batches}")

    def _upsert_to_bm25(self, documents: List[Document]):
        """Persists the raw documents to a local Pickle file for RAM-based BM25 search."""
        if not os.path.exists(settings.bm25_dir):
            os.makedirs(settings.bm25_dir)

        bm25_path = os.path.join(settings.bm25_dir, f"{self.collection_name}_bm25.pkl")
        existing_documents = []

        try:
            # 1. Load existing corpus to prevent complete overwrite
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    existing_documents = pickle.load(f)

            # 2. Track existing chunk_ids to avoid duplicating the same file
            existing_ids = {doc.metadata.get("chunk_id") for doc in existing_documents}
            new_documents = [
                doc
                for doc in documents
                if doc.metadata.get("chunk_id") not in existing_ids
            ]

            if not new_documents:
                logging.info(
                    f"No new documents to add for BM25 collection: {self.collection_name}"
                )
                return

            # 3. Append and save
            existing_documents.extend(new_documents)
            with open(bm25_path, "wb") as f:
                pickle.dump(existing_documents, f)

            logging.info(
                f"Lexical corpus persisted! Added {len(new_documents)} new documents."
            )
        except Exception as e:
            logging.error(f"Critical failure: Could not persist BM25 corpus: {e}")

    async def _a_upsert_to_bm25(self, documents: List[Document]):
        """Pickle is synchronous disk I/O, safely threaded and globally locked."""
        async with self._bm25_io_lock:
            await asyncio.to_thread(self._upsert_to_bm25, documents)

    async def _a_upsert_to_redis(self, documents: List[Document]):
        """
        Asynchronously creates a RediSearch index (if missing) and batch uploads
        documents to Upstash Redis for serverless BM25 execution.
        """
        from redis.commands.search.field import TextField, TagField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        # 2. Assign the 'redis' alias LAST
        import redis.asyncio as redis

        if not self.redis_client:
            raise ValueError("Redis client was not injected into the indexer.")

        index_name = f"idx:{self.collection_name}"
        prefix = f"doc:{self.collection_name}:"

        # 1. Ensure RediSearch Index Exists
        try:
            await self.redis_client.ft(index_name).info()
        except Exception:
            logging.info(f"Creating new RediSearch index: {index_name}")
            schema = (TextField("text"), TagField("chunk_id"), TextField("section"))
            definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
            await self.redis_client.ft(index_name).create_index(
                schema, definition=definition
            )

        # 2. Batch Upload via Pipeline (HSET inherently handles idempotent overwrites)
        pipeline = redis_client.pipeline(transaction=False)
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            doc_key = f"{prefix}{chunk_id}"

            mapping = {
                "text": doc.page_content,
                "chunk_id": chunk_id,
                "page_start": str(doc.metadata.get("page_start", "")),
                "page_end": str(doc.metadata.get("page_end", "")),
                "section": str(doc.metadata.get("section", "")),
            }
            pipeline.hset(doc_key, mapping=mapping)

        await pipeline.execute()

        logging.info(
            f"Lexical corpus persisted! {len(documents)} chunks upserted to Upstash Redis."
        )

    async def a_index_chunks(self, chunks, batch_size: int = 100):
        """Async orchestrator for the dual-indexing process."""
        if not chunks:
            logging.info("No chunks provided to index.")
            return

        documents, ids = self._prepare_documents(chunks)

        # 1. Handle Vector DB Upsert (with Idempotent Check)
        if await self._a_is_namespace_populated():
            logging.info(
                f"Namespace/Collection '{self.collection_name}' already contains vectors. Skipping Vector DB ingestion."
            )
        else:
            # Note: LangChain's PineconeVectorStore implements native asynchronous I/O.
            # Chroma lacks a native async client; LangChain's base VectorStore handles
            # this safely by routing aadd_documents to a background ThreadPoolExecutor.
            await self._a_upsert_to_vector_db(documents, ids, batch_size)

        # 2. Route Lexical DB Ingestion
        if settings.lexical_db_type == "redis":
            await self._a_upsert_to_redis(documents)
        else:
            await self._a_upsert_to_bm25(documents)

        logging.info("Async Indexing complete!")

    def get_retriever(self, k: int = 4):
        """Returns a standard LangChain retriever interface for pure semantic search."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})
