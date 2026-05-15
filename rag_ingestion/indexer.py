"""
indexer.py
-----------------------
Phase 4: Vector Embedding and Local Lexical Indexing.

This module acts as a factory for the vector database, supporting both
Chroma (local) and Pinecone (cloud). It also handles the persistence of
a local text corpus to allow for decoupled BM25 keyword search.

Key Features:
  - Idempotent Upserts: Uses your custom `chunk_id` so re-running the script
    updates existing chunks rather than duplicating them.
  - Namespace Checks: Checks if the vector namespace is already populated before hitting APIs.
  - BM25 Deduplication: Prevents bloating the local pickle file with duplicate documents.
"""

import asyncio
import os
import pickle
import logging
from typing import List, Tuple

# Conditional imports for Vector DB providers
from langchain_chroma import Chroma

try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import settings
from .models import PolicyChunk

logger = logging.getLogger(__name__)


class PolicyVectorStore:
    def __init__(self, collection_name: str, device: str = "cpu"):
        """
        Initializes the embedding model and connects to the configured Vector DB.

        Args:
            collection_name: The name of the collection (Chroma) or Namespace (Pinecone).
            device: 'cpu' or 'cuda' for embedding generation.
        """
        self.collection_name = collection_name

        # 1. Initialize Shared Embedding Model Dynamically
        logging.info(f"Initializing embedding model ({settings.embed_model_name})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True
            },  # Required for cosine similarity
        )

        # 2. Factory Initialization: Route to Pinecone or Chroma
        if settings.vector_db_type == "pinecone":
            if PineconeVectorStore is None:
                raise ImportError("langchain-pinecone is not installed.")

            logging.info(
                f"Connecting to Pinecone Index: {settings.pinecone_index_name} (Namespace: {collection_name})"
            )
            self.vector_store = PineconeVectorStore(
                index_name=settings.pinecone_index_name,
                embedding=self.embeddings,
                pinecone_api_key=settings.pinecone_api_key,
                namespace=collection_name,
            )
        else:
            logging.info(
                f"Connecting to local Chroma database at: {settings.chroma_dir}"
            )
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_dir,
            )

    def _get_bm25_path(self) -> str:
        """Returns the stable path for the local BM25 document corpus."""
        # Use the dedicated BM25 directory
        os.makedirs(settings.bm25_dir, exist_ok=True)
        # Create a specific pickle file for this collection
        return os.path.join(settings.bm25_dir, f"{self.collection_name}_bm25.pkl")

    async def _a_is_namespace_populated(self) -> bool:
        """Async check for namespace population."""
        return await asyncio.to_thread(self._is_namespace_populated)

    def _is_namespace_populated(self) -> bool:
        """
        Checks if the current namespace/collection already contains vectors.
        This prevents redundant embedding calculations and API calls.
        """
        try:
            if settings.vector_db_type == "pinecone":
                # Pinecone specific stats check
                stats = self.vector_store._index.describe_index_stats()
                namespaces = stats.get("namespaces", {})
                if (
                    self.collection_name in namespaces
                    and namespaces[self.collection_name]["vector_count"] > 0
                ):
                    return True
            else:
                # ChromaDB specific stats check
                count = self.vector_store._collection.count()
                if count > 0:
                    return True
        except Exception as e:
            logging.warning(f"Could not verify namespace stats, assuming empty: {e}")

        return False

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

    async def _a_upsert_to_vector_db(self, documents, ids, batch_size: int):
        """Asynchronous batch upsert to the configured Vector DB."""
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logging.info(
            f"Adding {len(documents)} chunks to {settings.vector_db_type.upper()} in {total_batches} batches..."
        )

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            # Use LangChain's native async add method
            await self.vector_store.aadd_documents(documents=batch_docs, ids=batch_ids)
            logging.info(f"  Processed batch {(i//batch_size) + 1}/{total_batches}")

    async def _a_upsert_to_bm25(self, documents):
        """Pickle is synchronous disk I/O, so we thread it."""
        async with self._bm25_io_lock:
            await asyncio.to_thread(self._upsert_to_bm25, documents)

    def _upsert_to_bm25(self, documents: List[Document]):
        """Persists the raw documents to a local Pickle file for BM25 hybrid search."""
        bm25_path = self._get_bm25_path()
        try:
            existing_documents = []
            existing_ids = set()

            # 1. Load existing corpus to prevent complete overwrite
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    existing_documents = pickle.load(f)
                    # Track existing chunk_ids to avoid duplicating the same file
                    existing_ids = {
                        doc.metadata.get("chunk_id")
                        for doc in existing_documents
                        if doc.metadata.get("chunk_id")
                    }

            # 2. Filter for strictly NEW documents
            new_documents = [
                doc
                for doc in documents
                if doc.metadata.get("chunk_id") not in existing_ids
            ]

            if not new_documents:
                logging.info(f"No new documents to add to BM25 corpus at: {bm25_path}")
                return

            # 3. Append and save
            existing_documents.extend(new_documents)
            with open(bm25_path, "wb") as f:
                pickle.dump(existing_documents, f)

            logging.info(
                f"Lexical corpus persisted! Added {len(new_documents)} new documents for local BM25 search at: {bm25_path}"
            )
        except Exception as e:
            logging.error(f"Critical failure: Could not persist BM25 corpus: {e}")

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

        # 2. Handle BM25 Persistence (Always run, relies on internal deduplication)
        await self._a_upsert_to_bm25(documents)

        logging.info("Async Indexing complete!")

    def get_retriever(self, k: int = 4):
        """
        Returns a standard LangChain retriever interface.
        Note: This is used for pure semantic retrieval.
        For Hybrid search, the RAGEngine will build a custom search engine.
        """
        return self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
