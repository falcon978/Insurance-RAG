"""
engine.py
---------
The central orchestrator for the Insurance RAG pipeline.

Responsibilities:
1. Manages provider-agnostic Vector DB connections (Chroma or Pinecone).
2. Maintains efficient BM25 index caching.
3. Manages Cross-Encoder reranking using concatenated contextual strings.
4. Handles unified single-pass adjudication and conversational state memory.
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Internal Module Imports
from config import settings
from rag.retriever import DocumentSearch
from rag.rerankers import ContextReranker
from rag.generator import ResponseGenerator
from rag.query_rewriter import get_structured_rewriter_chain

logger = logging.getLogger(__name__)


class InsuranceRAGEngine:
    def __init__(self, gemini_api_key: str):
        """
        Initializes the retrieval and generation engine with shared models,
        a localized BM25 cache, and the structured query translation chain.
        """
        logger.info(
            f"Booting RAG Engine (Vector Provider: {settings.vector_db_type.upper()})"
        )

        # 1. Initialize Shared Models Dynamically
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model_name,
            model_kwargs={"device": settings.hf_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.reranker = ContextReranker(
            model_name=settings.rerank_model_name,
            device=settings.hf_device,
        )
        self.generator = ResponseGenerator(
            api_key=gemini_api_key,
            model_name=settings.llm_model_name,
        )

        planner_llm = ChatGoogleGenerativeAI(
            model=settings.planner_model_name,
            temperature=settings.planner_temperature,
            api_key=gemini_api_key,
        )

        # 2. Initialize the Structured Output Translation Layer
        self.rewriter_chain = get_structured_rewriter_chain(planner_llm)

        # 3. Initialize Local Caching
        self._bm25_cache = {}

    def _load_bm25_retriever(self, collection_name: str) -> Optional[BM25Retriever]:
        """Loads the local text corpus for the BM25 lexical search branch."""
        if collection_name in self._bm25_cache:
            return self._bm25_cache[collection_name]

        corpus_path = os.path.join(settings.bm25_dir, f"{collection_name}_bm25.pkl")
        if not os.path.exists(corpus_path):
            logger.warning(
                f"BM25 index not found at {corpus_path}. Lexical search disabled."
            )
            return None

        try:
            with open(corpus_path, "rb") as f:
                docs = pickle.load(f)
            retriever = BM25Retriever.from_documents(docs)
            self._bm25_cache[collection_name] = retriever
            return retriever
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            return None

    def _get_search_engine(
        self,
        collection_name: str,
        retrieve_top_k: int,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
    ) -> DocumentSearch:
        """Connects to Chroma or Pinecone and attaches the BM25 local index."""
        if settings.vector_db_type == "pinecone":
            if not PineconeVectorStore:
                raise ImportError("pinecone-client is not installed.")
            vector_store = PineconeVectorStore(
                index_name=settings.pinecone_index_name,
                embedding=self.embeddings,
                namespace=collection_name,
                pinecone_api_key=settings.pinecone_api_key,
            )
        else:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_dir,
            )

        bm25_retriever = self._load_bm25_retriever(collection_name)

        return DocumentSearch(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            top_k=retrieve_top_k,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )

    def _format_policy_name(self, collection_name: str) -> str:
        """Cleans and formats raw collection names for UI presentation."""
        return collection_name.replace("insurance_", "").replace("_", " ").title()

    def query_single_policy(
        self,
        query: str,
        collection_name: str,
        history: Optional[List] = None,
        max_history_len: int = 6,
        **kwargs,
    ) -> str:
        """
        Runs the complete advanced retrieval and generation pipeline for a single policy.
        """
        active_history = history[-max_history_len:] if history else []
        ret_k = kwargs.get("retrieve_top_k", settings.default_retrieve_top_k)
        rerank_k = kwargs.get("rerank_top_k", settings.default_rerank_top_k)
        s_weight = kwargs.get("semantic_weight", settings.default_semantic_weight)
        l_weight = kwargs.get("lexical_weight", settings.default_lexical_weight)

        # 1. Generate Structured Intent
        structured_query = self.rewriter_chain.invoke({"query": query})

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
        fused_docs = search_engine.search(
            original_query=query, bm25_string=bm25_string, vector_string=vector_string
        )

        # 4. Cross-Encoder Reranking
        # Concatenates the lexical and semantic strings to provide comprehensive context to the Cross-Encoder.
        combined_rerank_string = f"{bm25_string} {vector_string}"
        best_docs = self.reranker.rerank(
            combined_rerank_string, fused_docs, top_k=rerank_k
        )

        # 5. Unified Single-Pass Generation
        # Passes the original query to the generator to preserve user tone and intent.
        return self.generator.generate_single_answer(
            query=query,
            docs=best_docs,
            policy_name=self._format_policy_name(collection_name),
            history=active_history,
        )

    def compare_policies(
        self,
        query: str,
        collection_a: str,
        collection_b: str,
        history: Optional[List] = None,
        max_history_len: int = 4,
        **kwargs,
    ) -> str:
        """
        Runs the comparison pipeline independently for two policies, ensuring strict isolation
        and deterministic semantic targeting.
        """
        active_history = history[-max_history_len:] if history else []
        ret_k = kwargs.get("retrieve_top_k", settings.default_retrieve_top_k)
        rerank_k = kwargs.get("rerank_top_k", settings.default_rerank_top_k)
        s_weight = kwargs.get("semantic_weight", settings.default_semantic_weight)
        l_weight = kwargs.get("lexical_weight", settings.default_lexical_weight)

        # 1. Generate Structured Intent (Executes once for consistency across both policies)
        structured_query = self.rewriter_chain.invoke({"query": query})

        # 2. Pre-Process Search Strings
        bm25_string = f"{query} {' '.join(structured_query.medical_terms)}".strip()
        vector_string = (
            f"{structured_query.canonical_query} "
            f"{' '.join(structured_query.expanded_terms)} "
            f"{' '.join(structured_query.exclusion_terms)} "
            f"{' '.join(structured_query.policy_sections)}"
        ).strip()

        # 3. Retrieve via Asymmetric Ensemble (Executes independently per policy)
        search_engine_a = self._get_search_engine(
            collection_a, ret_k, s_weight, l_weight
        )
        fused_a = search_engine_a.search(query, bm25_string, vector_string)

        search_engine_b = self._get_search_engine(
            collection_b, ret_k, s_weight, l_weight
        )
        fused_b = search_engine_b.search(query, bm25_string, vector_string)

        # 4. Cross-Encoder Reranking
        combined_rerank_string = f"{bm25_string} {vector_string}"
        best_a = self.reranker.rerank(
            combined_rerank_string,
            fused_a,
            top_k=rerank_k,
        )
        best_b = self.reranker.rerank(
            combined_rerank_string,
            fused_b,
            top_k=rerank_k,
        )

        # 5. Unified Single-Pass Comparative Generation
        return self.generator.generate_comparison(
            query=query,
            docs_a=best_a,
            name_a=self._format_policy_name(collection_a),
            docs_b=best_b,
            name_b=self._format_policy_name(collection_b),
            history=active_history,
        )
