"""
rag/retriever.py
----------------
Executes Asymmetric Ensemble Retrieval.
Orchestrates parallel multi-string searches across Vector and Lexical databases
and merges them using Two-Stage Reciprocal Rank Fusion via asynchronous execution.
"""

import re
import logging
import asyncio
from typing import List, Optional
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.utils import fuse_multi_query_results, fuse_weighted_results

logger = logging.getLogger(__name__)


class UpstashRediSearchRetriever:
    """
    Custom LangChain-compatible retriever that executes full-text BM25
    searches natively on an Upstash Redis database via RediSearch.
    This offloads heavy tokenization and BM25 math from the Python server.
    """

    def __init__(self, redis_url: str, collection_name: str, k: int = 15):
        import redis.asyncio as redis

        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.index_name = f"idx:{collection_name}"
        self.k = k

    async def ainvoke(self, query: str) -> List[Document]:
        from redis.commands.search.query import Query

        try:
            # RediSearch syntax reserves punctuation; strip it for safe token matching
            clean_query = re.sub(r"[^\w\s]", " ", query).strip()
            if not clean_query:
                return []

            # Execute native BM25 search on the Upstash server
            q = Query(clean_query).paging(0, self.k)
            res = await self.redis_client.ft(self.index_name).search(q)

            docs = []
            for doc in res.docs:
                # RediSearch stores metadata as strings; reconstruct the LangChain Document
                metadata = {
                    key: val
                    for key, val in doc.__dict__.items()
                    if key not in ["id", "payload", "text"]
                }
                docs.append(
                    Document(page_content=getattr(doc, "text", ""), metadata=metadata)
                )

            return docs
        except Exception as e:
            logger.error(f"Upstash RediSearch query failed: {e}")
            return []
        finally:
            await self.redis_client.aclose()


class DocumentSearch:
    """
    Orchestrates Hybrid Retrieval by fusing results from semantic and lexical searches.
    """

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
            bm25_retriever: Optional BM25 retriever (local or Upstash) for lexical search.
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

        # Upstash custom retriever handles 'k' internally during init,
        # but LangChain's local BM25 requires setting it as an attribute
        if hasattr(self.bm25_retriever, "k"):
            self.bm25_retriever.k = self.top_k

    async def a_search(
        self, original_query: str, bm25_string: str, vector_string: str
    ) -> List[Document]:
        """
        Executes the Asymmetric Ensemble Retrieval Pipeline asynchronously and manually calculates the
        Reciprocal Rank Fusion (RRF) scores to combine the contexts.
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

        # Stage 2: Weighted fusion of the semantic super-set with the lexical track
        final_fused = fuse_weighted_results(
            list_a=semantic_fused,
            list_b=docs_bm25,
            weight_a=self.semantic_weight,
            weight_b=self.lexical_weight,
        )

        return final_fused
