"""
engine.py
---------
The central orchestrator for the Insurance RAG pipeline. 
"""

import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.retriever import DocumentSearch
from rag.rerankers import ContextReranker
from rag.generator import ResponseGenerator

logger = logging.getLogger(__name__)

class InsuranceRAGEngine:
    def __init__(self, gemini_api_key: str, chroma_dir: str = "./chroma_data"):
        logger.info("Booting up shared RAG Engine components...")
        self.chroma_dir = chroma_dir
        
        # Initialize Shared Models
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        self.reranker = ContextReranker(device="cpu")
        self.generator = ResponseGenerator(api_key=gemini_api_key)

    # FIX: Expose retrieve_top_k here so it gets passed to the retriever
    def _get_search_engine(self, collection_name: str, retrieve_top_k: int) -> DocumentSearch:
        """Dynamically connects to a Chroma collection and returns a Hybrid Search engine."""
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.chroma_dir
        )
        return DocumentSearch(vector_store, strategy="hybrid", top_k=retrieve_top_k)

    def _format_policy_name(self, collection_name: str) -> str:
        clean_name = collection_name.replace("insurance_", "").replace("_", " ")
        return clean_name.title()

    # FIX: Expose retrieve_top_k and rerank_top_k as arguments
    def query_single_policy(self, query: str, collection_name: str, retrieve_top_k: int = 15, rerank_top_k: int = 3) -> str:
        """Runs the 3-step pipeline for a dynamically selected policy."""
        logger.info(f"Processing single policy query for: {collection_name}")
        
        # Pass retrieve_top_k to the engine builder
        search_engine = self._get_search_engine(collection_name, retrieve_top_k)
        policy_name = self._format_policy_name(collection_name)
        
        # 1. Broad Retrieval
        broad_docs = search_engine.search(query)
        # 2. Reranking (Pass rerank_top_k)
        best_docs = self.reranker.rerank(query, broad_docs, top_k=rerank_top_k)
        # 3. Generation
        return self.generator.generate_single_answer(query, best_docs, policy_name)

    # FIX: Expose retrieve_top_k and rerank_top_k as arguments
    def compare_policies(self, query: str, collection_a: str, collection_b: str, retrieve_top_k: int = 15, rerank_top_k: int = 3) -> str:
        """Runs the pipeline simultaneously across two dynamically selected policies."""
        logger.info(f"Processing comparison: {collection_a} vs {collection_b}")
        
        # Pass retrieve_top_k to both engines
        engine_a = self._get_search_engine(collection_a, retrieve_top_k)
        engine_b = self._get_search_engine(collection_b, retrieve_top_k)
        
        # Retrieve & Rerank A
        broad_a = engine_a.search(query)
        best_a = self.reranker.rerank(query, broad_a, top_k=rerank_top_k)
        
        # Retrieve & Rerank B
        broad_b = engine_b.search(query)
        best_b = self.reranker.rerank(query, broad_b, top_k=rerank_top_k)
        
        # Generate
        return self.generator.generate_comparison(
            query=query, 
            docs_a=best_a, policy_name_a=self._format_policy_name(collection_a),
            docs_b=best_b, policy_name_b=self._format_policy_name(collection_b)
        )