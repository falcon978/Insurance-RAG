"""
engine.py
---------
The central orchestrator for the Insurance RAG pipeline.

Responsibilities:
1. Orchestrates the retrieval and generation workflow.
2. Delegates database initialization to dedicated factories.
3. Executes Cross-Encoder reranking using concatenated contextual strings.
4. Handles unified single-pass adjudication and conversational state memory.
"""

import asyncio
import logging
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Internal Module Imports
from config import settings
from rag.factories import VectorStoreFactory, LexicalStoreFactory
from rag.retriever import DocumentSearch
from rag.rerankers import ContextReranker
from rag.generator import ResponseGenerator
from rag.query_rewriter import get_structured_rewriter_chain

logger = logging.getLogger(__name__)


class InsuranceRAGEngine:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    Initializes shared models and delegates data fetching to the retrieval layer.
    """

    def __init__(self, gemini_api_key: str):
        """
        Initializes the retrieval and generation engine with shared models
        and the structured query translation chain.
        """
        logger.info(
            f"Booting RAG Engine (Vector Provider: {settings.vector_db_type.upper()} | Lexical: {settings.lexical_db_type.upper()})"
        )

        # 1. Initialize Shared Models Dynamically
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model_name,
            model_kwargs={"device": settings.hf_device},
            encode_kwargs={
                "normalize_embeddings": True
            },  # Enforces L2 Normalization for dot-product compatibility
        )
        self.reranker = ContextReranker(
            model_name=settings.rerank_model_name, device=settings.hf_device
        )
        self.generator = ResponseGenerator(
            api_key=gemini_api_key, model_name=settings.llm_model_name
        )

        # 2. Initialize the Structured Output Translation Layer
        planner_llm = ChatGoogleGenerativeAI(
            model=settings.planner_model_name,
            temperature=settings.planner_temperature,
            api_key=gemini_api_key,
        )

        # 3. Initialize the Structured Output Translation Layer
        self.rewriter_chain = get_structured_rewriter_chain(planner_llm)

    def _get_search_engine(
        self,
        collection_name: str,
        retrieve_top_k: int,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
    ) -> DocumentSearch:
        """
        Utilizes factories to instantiate the required databases and wraps them in the Orchestrator.
        """
        # 1. Delegate Vector Database creation to the Factory
        vector_store = VectorStoreFactory.get_vector_store(
            collection_name=collection_name, embeddings=self.embeddings
        )

        # 2. Delegate Lexical Database creation to the Factory
        bm25_retriever = LexicalStoreFactory.get_lexical_retriever(
            collection_name=collection_name, top_k=retrieve_top_k
        )

        # 3. Return the Orchestrator
        return DocumentSearch(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            top_k=retrieve_top_k,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )

    def _format_policy_name(self, collection_name: str) -> str:
        """Cleans and formats raw collection names for UI presentation."""
        name = collection_name.replace("insurance_", "").replace("_", " ").title()
        return name

    async def a_query_single_policy(
        self,
        query: str,
        collection_name: str,
        history: Optional[List] = None,
        max_history_len: int = 6,
        **kwargs,
    ) -> str:
        """Orchestrates the retrieval and generation pipeline for a single policy."""
        active_history = history[-max_history_len:] if history else []
        ret_k = kwargs.get("retrieve_top_k", settings.default_retrieve_top_k)
        rerank_k = kwargs.get("rerank_top_k", settings.default_rerank_top_k)
        s_weight = kwargs.get("semantic_weight", settings.default_semantic_weight)
        l_weight = kwargs.get("lexical_weight", settings.default_lexical_weight)

        # 1. Generate Structured Intent Representation for Retrieval Optimization
        structured_query = await self.rewriter_chain.ainvoke({"query": query})

        # 2. Pre-Process Search Strings
        bm25_string = f"{query} {' '.join(structured_query.medical_terms)}".strip()
        vector_string = (
            f"{structured_query.canonical_query} "
            f"{' '.join(structured_query.expanded_terms)} "
            f"{' '.join(structured_query.exclusion_terms)} "
            f"{' '.join(structured_query.policy_sections)}"
        ).strip()

        logger.info(f"[Retrieval Target] {collection_name}")
        logger.info(f"[BM25 Query] '{bm25_string}'")
        logger.info(f"[Vector Query] '{vector_string}'")

        # 3. Retrieve via Asymmetric Ensemble
        search_engine = self._get_search_engine(
            collection_name, ret_k, s_weight, l_weight
        )
        fused_docs = await search_engine.a_search(
            original_query=query, bm25_string=bm25_string, vector_string=vector_string
        )

        # 4. Cross-Encoder Reranking
        # Concatenates the lexical and semantic strings to provide comprehensive context to the Cross-Encoder.
        combined_rerank_string = f"{bm25_string} {vector_string}"

        best_docs = await self.reranker.a_rerank(
            query=combined_rerank_string, documents=fused_docs, top_k=rerank_k
        )

        # 5. Unified Single-Pass Generation
        # Passes the original query to the generator to preserve user tone and intent.
        return await self.generator.a_generate_single_answer(
            query=query,
            docs=best_docs,
            policy_name=self._format_policy_name(collection_name),
            history=active_history,
        )

    async def a_compare_policies(
        self,
        query: str,
        collection_a: str,
        collection_b: str,
        history: Optional[List] = None,
        max_history_len: int = 4,
        **kwargs,
    ) -> str:
        """Orchestrates the independent comparative retrieval and generation pipeline."""
        active_history = history[-max_history_len:] if history else []
        ret_k = kwargs.get("retrieve_top_k", settings.default_retrieve_top_k)
        rerank_k = kwargs.get("rerank_top_k", settings.default_rerank_top_k)
        s_weight = kwargs.get("semantic_weight", settings.default_semantic_weight)
        l_weight = kwargs.get("lexical_weight", settings.default_lexical_weight)

        # 1. Generate Structured Intent (Executes once for consistency across both policies)
        structured_query = await self.rewriter_chain.ainvoke({"query": query})

        # 2. Pre-Process Search Strings
        bm25_string = f"{query} {' '.join(structured_query.medical_terms)}".strip()
        vector_string = (
            f"{structured_query.canonical_query} "
            f"{' '.join(structured_query.expanded_terms)} "
            f"{' '.join(structured_query.exclusion_terms)} "
            f"{' '.join(structured_query.policy_sections)}"
        ).strip()

        logger.info(f"[Comparative Retrieval] '{collection_a}' vs '{collection_b}'")
        logger.info(f"[BM25 Query] '{bm25_string}'")
        logger.info(f"[Vector Query] '{vector_string}'")

        # 3. Retrieve via Asymmetric Ensemble (Executes independently per policy)
        search_engine_a = self._get_search_engine(
            collection_a, ret_k, s_weight, l_weight
        )
        search_engine_b = self._get_search_engine(
            collection_b, ret_k, s_weight, l_weight
        )

        # Executes retrieval for both collections concurrently
        fused_a, fused_b = await asyncio.gather(
            search_engine_a.a_search(query, bm25_string, vector_string),
            search_engine_b.a_search(query, bm25_string, vector_string),
        )

        # 4. Cross-Encoder Reranking
        combined_rerank_string = f"{bm25_string} {vector_string}"

        # Delegates CPU-bound reranking tasks to background threads concurrently
        best_a, best_b = await asyncio.gather(
            self.reranker.a_rerank(
                query=combined_rerank_string, documents=fused_a, top_k=rerank_k
            ),
            self.reranker.a_rerank(
                query=combined_rerank_string, documents=fused_b, top_k=rerank_k
            ),
        )

        # 5. Unified Single-Pass Comparative Generation
        return await self.generator.a_generate_comparison(
            query=query,
            docs_a=best_a,
            name_a=self._format_policy_name(collection_a),
            docs_b=best_b,
            name_b=self._format_policy_name(collection_b),
            history=active_history,
        )
