"""
engine.py
---------
The central orchestrator for the Insurance RAG pipeline.

Responsibilities:
1. Manages provider-agnostic Vector DB connections (Chroma or Pinecone).
2. Maintains efficient BM25 index caching.
3. Implements Asymmetric Ensemble Retrieval (Orthogonal Hybrid Search).
4. Executes Two-Stage Reciprocal Rank Fusion (RRF) for lexical/semantic balancing.
5. Manages Cross-Encoder reranking using concatenated contextual strings.
6. Handles unified single-pass adjudication and conversational state memory.
"""

import os
import pickle
import logging
import concurrent.futures
from typing import List, Optional

from langchain_chroma import Chroma

try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Internal Module Imports
from config import settings
from rag.retriever import DocumentSearch
from rag.rerankers import ContextReranker
from rag.generator import ResponseGenerator
from rag.query_rewriter import get_structured_rewriter_chain
from rag.utils import fuse_multi_query_results, fuse_weighted_results

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

        # 2. Initialize the Structured Output Translation Layer
        self.rewriter_chain = get_structured_rewriter_chain(self.generator.llm)

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
        self, collection_name: str, retrieve_top_k: int
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
            strategy="hybrid",
            top_k=retrieve_top_k,
        )

    def _format_policy_name(self, collection_name: str) -> str:
        """Cleans and formats raw collection names for UI presentation."""
        return collection_name.replace("insurance_", "").replace("_", " ").title()

    def _dual_track_retrieve(
        self,
        query: str,
        bm25_search_string: str,
        vector_search_string: str,
        collection_name: str,
        top_k: int,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
    ) -> List[Document]:
        """
        Executes the Asymmetric Ensemble Retrieval Pipeline.
        Accepts pre-formatted search strings to ensure deterministic execution
        and decouple string processing from core retrieval logic.

        Process:
        1. Runs 3 parallel searches (BM25 Lexical, Vector Original, Vector Dense).
        2. Applies Two-Stage Reciprocal Rank Fusion (Lexical/Semantic Split).

        Args:
            query (str): The original user query.
            bm25_search_string (str): The string optimized for sparse lexical retrieval.
            vector_search_string (str): The string optimized for dense semantic retrieval.
            collection_name (str): The target vector DB collection/namespace.
            top_k (int): Number of documents to retrieve per search path.
            semantic_weight (float): The weight assigned to the semantic tracks in the final fusion.
            lexical_weight (float): The weight assigned to the lexical tracks in the final fusion.

        Returns:
            List[Document]: The fused and ranked document list.
        """
        search_engine = self._get_search_engine(collection_name, top_k)
        vector_retriever = search_engine.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        bm25_retriever = search_engine.bm25_retriever

        logger.info(f"[Retrieval Target] {collection_name}")
        logger.info(f"[BM25 Query] '{bm25_search_string}'")
        logger.info(f"[Vector Query] '{vector_search_string}'")

        # Parallel Database Execution to minimize sequential latency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Path A: Vector search on the original query (Semantic Anchor)
            future_vec_orig = executor.submit(vector_retriever.invoke, query)
            # Path B: Vector search on the dense semantic string (Bridging Vocabulary)
            future_vec_legal = executor.submit(
                vector_retriever.invoke, vector_search_string
            )

            # Path C: BM25 search on the lexical string (Original intent + Clinical Terms)
            if bm25_retriever:
                future_bm25 = executor.submit(bm25_retriever.invoke, bm25_search_string)

            docs_vec_orig = future_vec_orig.result()
            docs_vec_legal = future_vec_legal.result()
            docs_bm25 = future_bm25.result() if bm25_retriever else []

        # Two-Stage Reciprocal Rank Fusion
        # Stage 1: Standard unweighted fusion of the two Semantic tracks
        semantic_fused = fuse_multi_query_results([docs_vec_orig, docs_vec_legal])

        # Stage 2: Weighted fusion of Semantic vs. Lexical tracks
        final_fused = fuse_weighted_results(
            list_a=semantic_fused,
            list_b=docs_bm25,
            weight_a=semantic_weight,
            weight_b=lexical_weight,
        )

        return final_fused

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

        # 3. Dual-Track Retrieval
        fused_docs = self._dual_track_retrieve(
            query,
            bm25_string,
            vector_string,
            collection_name,
            kwargs.get("retrieve_top_k", 15),
            semantic_weight=kwargs.get("semantic_weight", 0.5),
            lexical_weight=kwargs.get("lexical_weight", 0.5),
        )

        # 4. Cross-Encoder Reranking
        # Concatenates the lexical and semantic strings to provide comprehensive context to the Cross-Encoder.
        combined_rerank_string = f"{bm25_string} {vector_string}"
        best_docs = self.reranker.rerank(
            combined_rerank_string, fused_docs, top_k=kwargs.get("rerank_top_k", 5)
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
        ret_k = kwargs.get("retrieve_top_k", 15)
        rerank_k = kwargs.get("rerank_top_k", 5)

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

        # 3. Dual-Track Retrieval (Executes independently per policy)
        fused_a = self._dual_track_retrieve(
            query, bm25_string, vector_string, collection_a, ret_k
        )
        fused_b = self._dual_track_retrieve(
            query, bm25_string, vector_string, collection_b, ret_k
        )

        # 4. Cross-Encoder Reranking
        s_weight = kwargs.get("semantic_weight", 0.5)
        l_weight = kwargs.get("lexical_weight", 0.5)
        combined_rerank_string = f"{bm25_string} {vector_string}"
        best_a = self.reranker.rerank(
            combined_rerank_string,
            fused_a,
            top_k=rerank_k,
            semantic_weight=s_weight,
            lexical_weight=l_weight,
        )
        best_b = self.reranker.rerank(
            combined_rerank_string,
            fused_b,
            top_k=rerank_k,
            semantic_weight=s_weight,
            lexical_weight=l_weight,
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
