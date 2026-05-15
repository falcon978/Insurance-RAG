"""
rag/retriever.py
----------------
Executes Asymmetric Ensemble Retrieval.
Orchestrates parallel multi-string searches across Vector and Lexical databases
and merges them using Two-Stage Reciprocal Rank Fusion via asynchronous execution.
"""

import logging
import asyncio
from typing import List, Optional
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

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

    async def a_search(
        self, original_query: str, bm25_string: str, vector_string: str
    ) -> List[Document]:
        """
        Executes the Asymmetric Ensemble Retrieval Pipeline asynchronously.

        Args:
            original_query (str): The original user query.
            bm25_string (str): The string optimized for sparse lexical retrieval.
            vector_string (str): The string optimized for dense semantic retrieval.

        Returns:
            List[Document]: The fused and ranked document list.
        """
        tasks = [
            self.vector_retriever.ainvoke(original_query),
            self.vector_retriever.ainvoke(vector_string),
        ]

        if self.bm25_retriever:
            tasks.append(self.bm25_retriever.ainvoke(bm25_string))

        results = await asyncio.gather(*tasks)

        docs_vec_orig = results[0]
        docs_vec_legal = results[1]
        docs_bm25 = results[2] if self.bm25_retriever else []

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
