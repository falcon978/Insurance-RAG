"""
rag/retriever.py
----------------
Executes Asymmetric Ensemble Retrieval.
Orchestrates parallel multi-string searches across Vector and Lexical databases
and merges them using Two-Stage Reciprocal Rank Fusion.
"""

import logging
import concurrent.futures
from typing import List, Optional
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Import fusion utilities for merging results
from rag.utils import fuse_multi_query_results, fuse_weighted_results

logger = logging.getLogger(__name__)


class DocumentSearch:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: Optional[BM25Retriever] = None,
        top_k: int = 15,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
    ):
        """
        Initializes the search engine with a provider-agnostic vector store
        and an optional lexical retriever.

        Args:
            vector_store: Any LangChain VectorStore (Chroma, Pinecone, etc.).
            bm25_retriever: A pre-built BM25 retriever cached by the Engine.
            top_k: Number of documents to retrieve per retrieval branch.
            semantic_weight: Weight given to semantic (vector) results during final fusion.
            lexical_weight: Weight given to lexical (BM25) results during final fusion.
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

        # Pre-build the underlying LangChain retrievers
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        if self.bm25_retriever:
            self.bm25_retriever.k = self.top_k

    def search(
        self, original_query: str, bm25_string: str, vector_string: str
    ) -> List[Document]:
        """
        Executes the Asymmetric Ensemble Retrieval Pipeline.

        Process:
        1. Runs 3 parallel searches (BM25 Lexical, Vector Original, Vector Dense).
        2. Applies Two-Stage Reciprocal Rank Fusion (Lexical/Semantic Split).

        Args:
            original_query (str): The original user query.
            bm25_string (str): The string optimized for sparse lexical retrieval.
            vector_string (str): The string optimized for dense semantic retrieval.

        Returns:
            List[Document]: The fused and ranked document list.
        """
        # Parallel Database Execution to minimize sequential latency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Path A: Vector search on the original query (Semantic Anchor)
            future_vec_orig = executor.submit(
                self.vector_retriever.invoke, original_query
            )

            # Path B: Vector search on the dense semantic string (Bridging Vocabulary)
            future_vec_legal = executor.submit(
                self.vector_retriever.invoke, vector_string
            )

            # Path C: BM25 search on the lexical string (Original intent + Clinical Terms)
            if self.bm25_retriever:
                future_bm25 = executor.submit(self.bm25_retriever.invoke, bm25_string)

            docs_vec_orig = future_vec_orig.result()
            docs_vec_legal = future_vec_legal.result()
            docs_bm25 = future_bm25.result() if self.bm25_retriever else []

        # Two-Stage Reciprocal Rank Fusion
        # Stage 1: Standard unweighted fusion of the two Semantic tracks
        semantic_fused = fuse_multi_query_results([docs_vec_orig, docs_vec_legal])

        # Stage 2: Weighted fusion of Semantic vs. Lexical tracks
        final_fused = fuse_weighted_results(
            list_a=semantic_fused,
            list_b=docs_bm25,
            weight_a=self.semantic_weight,
            weight_b=self.lexical_weight,
        )

        return final_fused
