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
  - Metadata Flattening: ChromaDB strictly requires metadata values to be strings, 
    integers, floats, or booleans. We flatten the nested dicts here.
"""

import os
import pickle
import logging
from typing import List

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
        
        # 1. Initialize Shared Embedding Model
        logging.info(f"Initializing embedding model (BAAI/bge-large-en-v1.5)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': device}, 
            encode_kwargs={'normalize_embeddings': True} # Required for cosine similarity
        )
        
        # 2. Factory Initialization: Route to Pinecone or Chroma
        if settings.vector_db_type == "pinecone":
            if PineconeVectorStore is None:
                raise ImportError("langchain-pinecone is not installed.")
            
            logging.info(f"Connecting to Pinecone Index: {settings.pinecone_index_name} (Namespace: {collection_name})")
            self.vector_store = PineconeVectorStore(
                index_name=settings.pinecone_index_name,
                embedding=self.embeddings,
                pinecone_api_key=settings.pinecone_api_key,
                namespace=collection_name
            )
        else:
            logging.info(f"Connecting to local Chroma database at: {settings.chroma_dir}")
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_dir
            )

    def _get_bm25_path(self) -> str:
        """Returns the stable path for the local BM25 document corpus."""
        # Use the dedicated BM25 directory
        os.makedirs(settings.bm25_dir, exist_ok=True)
        # Create a specific pickle file for this collection
        return os.path.join(settings.bm25_dir, f"{self.collection_name}_bm25.pkl")

    def index_chunks(self, chunks: List[PolicyChunk], batch_size: int = 100):
        """
        Dual-indexes chunks: 
        1. Upserts embeddings to the Vector DB.
        2. Persists the Document corpus locally for the BM25 index.
        """
        if not chunks:
            logging.info("No chunks provided to index.")
            return

        documents = []
        ids = []

        for chunk in chunks:
            # Flatten metadata for database cross-compatibility
            flat_metadata = {
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
                "char_count": chunk.metadata.get("char_count", 0)
            }

            doc = Document(page_content=chunk.text, metadata=flat_metadata)
            documents.append(doc)
            ids.append(chunk.chunk_id)

        # A. Batch Upsert to Vector Store
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logging.info(f"Adding {len(documents)} chunks to {settings.vector_db_type.upper()} in {total_batches} batches...")

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            logging.info(f"  Processed batch {(i//batch_size) + 1}/{total_batches}")

        # B. Decoupled BM25 Corpus Persistence
        bm25_path = self._get_bm25_path()
        try:
            # 1. Load existing corpus if it exists to prevent overwriting previous PDFs
            existing_documents = []
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    existing_documents = pickle.load(f)
            
            # 2. Append new documents
            existing_documents.extend(documents)

            # 3. Save combined corpus
            with open(bm25_path, "wb") as f:
                pickle.dump(existing_documents, f)
            logging.info(f"Lexical corpus persisted for local BM25 search at: {bm25_path}")
        except Exception as e:
            logging.error(f"Critical failure: Could not persist BM25 corpus: {e}")

        logging.info("Indexing complete!")

    def get_retriever(self, k: int = 4):
        """
        Returns a standard LangChain retriever interface. 
        Note: This is used for pure semantic retrieval. 
        For Hybrid search, the RAGEngine will build a custom search engine.
        """
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )