"""
indexer.py
----------
Phase 4: Vector Embedding and Database Storage.

Takes the RAG-ready PolicyChunk objects, embeds them using an open-source 
HuggingFace model, and stores them persistently in ChromaDB.

Key Features:
  - Idempotent Upserts: Uses your custom `chunk_id` so re-running the script 
    updates existing chunks rather than duplicating them.
  - Metadata Flattening: ChromaDB strictly requires metadata values to be strings, 
    integers, floats, or booleans. We flatten the nested dicts here.
"""

from typing import List
import os
import logging

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .models import PolicyChunk

logger = logging.getLogger(__name__)


class PolicyVectorStore:
    """
    Handles embedding and storing policy chunks into a local Chroma vector database.
    """

    def __init__(
        self, 
        persist_directory: str = "./chroma_data", 
        collection_name: str = "insurance_policies",
        device: str = "cpu" # Change to "cuda" if you have an Nvidia GPU
    ):
        logging.info(f"Initializing embedding model (this may take a moment)...")
        # Using the BAAI/bge-large-en-v1.5 model as discussed
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': device}, 
            encode_kwargs={'normalize_embeddings': True} # Crucial for cosine similarity
        )
        
        logging.info(f"Connecting to local Chroma database at '{persist_directory}'...")
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def index_chunks(self, chunks: List[PolicyChunk], batch_size: int = 100):
        """
        Converts PolicyChunks to LangChain Documents and upserts them in batches.
        """
        if not chunks:
            logging.info("No chunks provided to index.")
            return

        documents = []
        ids = []

        for chunk in chunks:
            # 1. Flatten the metadata for ChromaDB
            # Chroma rejects nested dictionaries or lists, so we pull everything 
            # to the top level.
            flat_metadata = {
                "source_file": chunk.source_file,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "sub_section": chunk.sub_section,
                "heading": chunk.heading,
                "token_estimate": chunk.token_estimate,
                # Extracting from the internal metadata dict
                "has_table": chunk.metadata.get("has_table", False),
                "has_list": chunk.metadata.get("has_list", False),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "char_count": chunk.metadata.get("char_count", 0)
            }

            # 2. Create the LangChain Document object
            doc = Document(
                page_content=chunk.text, # This includes your injected lineage prefix!
                metadata=flat_metadata
            )
            
            documents.append(doc)
            ids.append(chunk.chunk_id)

        # 3. Batch Upsert to avoid memory limits
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logging.info(f"Adding {len(documents)} chunks to the database in {total_batches} batches...")

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            
            # add_documents acts as an upsert if IDs are provided
            self.vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            logging.info(f"  Processed batch {(i//batch_size) + 1}/{total_batches}")

        logging.info("Indexing complete!")
        
    def get_retriever(self, k: int = 4):
        """
        Returns a LangChain retriever interface for the RAG pipeline.
        """
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )