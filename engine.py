"""
engine.py
---------
The central orchestrator for the Insurance RAG pipeline. 

It manages:
1. Provider-Agnostic Vector DB connections (Chroma or Pinecone).
2. Efficient BM25 Index Caching.
3. The 3-Step RAG Pipeline: Hybrid Retrieval -> Reranking -> Adjudication.
4. Stateful conversational memory with a sliding window
"""

import os
import pickle
import logging
from typing import List, Optional

from langchain_chroma import Chroma
try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# Internal Module Imports
from config import settings
from rag.retriever import DocumentSearch
from rag.rerankers import ContextReranker
from rag.generator import ResponseGenerator

logger = logging.getLogger(__name__)

class InsuranceRAGEngine:
    def __init__(self, gemini_api_key: str):
        """
        Initializes the engine with shared components and an empty BM25 cache.
        """
        logger.info(f"Booting RAG Engine (Vector Provider: {settings.vector_db_type.upper()})")
        
        # 1. Initialize Shared Models
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': settings.hf_device}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        self.reranker = ContextReranker(device=settings.hf_device)
        self.generator = ResponseGenerator(api_key=gemini_api_key)
        
        # Cache to store BM25 index in memory to avoid rebuilding every query
        self._bm25_cache = {}

    def _load_bm25_retriever(self, collection_name: str) -> Optional[BM25Retriever]:
        """Loads the local text corpus for the BM25 lexical search branch."""
        if collection_name in self._bm25_cache:
            return self._bm25_cache[collection_name]

        # Lexical corpus is always local for low-latency keyword search
        corpus_path = os.path.join(settings.bm25_dir, f"{collection_name}_bm25.pkl")
        
        if not os.path.exists(corpus_path):
            logger.warning(f"BM25 corpus missing at {corpus_path}. Performance may degrade.")
            return None

        try:
            with open(corpus_path, "rb") as f:
                docs = pickle.load(f)
            
            # Initializing BM25 is CPU-intensive; we do this once per policy
            retriever = BM25Retriever.from_documents(docs)
            self._bm25_cache[collection_name] = retriever
            logger.info(f"Loaded BM25 index for {collection_name} into cache.")
            return retriever
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            return None

    def _get_search_engine(self, collection_name: str, retrieve_top_k: int) -> DocumentSearch:
        """Factory method to connect to the configured Vector DB (Chroma or Pinecone)."""
        if settings.vector_db_type == "pinecone":
            vector_store = PineconeVectorStore(
                index_name=settings.pinecone_index_name,
                embedding=self.embeddings,
                namespace=collection_name,
                pinecone_api_key=settings.pinecone_api_key
            )
        else:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_dir
            )
        
        # Fetch the local BM25 retriever
        bm25_retriever = self._load_bm25_retriever(collection_name)
        
        return DocumentSearch(
            vector_store=vector_store, 
            bm25_retriever=bm25_retriever, 
            strategy="hybrid", 
            top_k=retrieve_top_k
        )

    def _format_policy_name(self, collection_name: str) -> str:
        """Converts raw keys (e.g. 'insurance_hdfc_ergo') to clean titles."""
        return collection_name.replace("insurance_", "").replace("_", " ").title()

    def query_single_policy(
        self, 
        query: str, 
        collection_name: str, 
        history: Optional[List] = None, 
        max_history_len: int = 6, # Keeps last 3 Human-AI turns
        **kwargs
    ) -> str:
        """3-Step Stateful Pipeline: Retrieve -> Rerank -> Generate with Sliding Window Memory."""
        
        # SLIDING WINDOW: Trim history before sending to the LLM
        active_history = (history[-max_history_len:] if history else [])
        
        search_engine = self._get_search_engine(collection_name, kwargs.get('retrieve_top_k', 15))
        
        # 1. Hybrid Retrieval
        broad_docs = search_engine.search(query)
        
        # 2. Reranking
        best_docs = self.reranker.rerank(query, broad_docs, top_k=kwargs.get('rerank_top_k', 3))
        
        # 3. Two-Pass Generation with Trimmed History
        return self.generator.generate_single_answer(
            query=query, 
            docs=best_docs, 
            policy_name=self._format_policy_name(collection_name),
            history=active_history
        )

    def compare_policies(
        self, 
        query: str, 
        collection_a: str, 
        collection_b: str, 
        history: Optional[List] = None, 
        max_history_len: int = 4,
        **kwargs
    ) -> str:
        """Runs the comparison pipeline with conversational state."""
        
        active_history = (history[-max_history_len:] if history else [])
        
        eng_a = self._get_search_engine(collection_a, kwargs.get('retrieve_top_k', 15))
        eng_b = self._get_search_engine(collection_b, kwargs.get('retrieve_top_k', 15))
        
        # Process Policy A
        best_a = self.reranker.rerank(query, eng_a.search(query), top_k=kwargs.get('rerank_top_k', 3))
        # Process Policy B
        best_b = self.reranker.rerank(query, eng_b.search(query), top_k=kwargs.get('rerank_top_k', 3))
        
        # Generate Comparative Decision
        return self.generator.generate_comparison(
            query=query, 
            docs_a=best_a, name_a=self._format_policy_name(collection_a),
            docs_b=best_b, name_b=self._format_policy_name(collection_b),
            history=active_history
        )