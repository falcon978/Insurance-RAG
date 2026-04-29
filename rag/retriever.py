"""
rag/retriever.py
----------------
Executes Hybrid Search by orchestrating Vector and Lexical retrievers.
This version is decoupled from the Vector DB's internal storage and 
relies on a pre-initialized BM25 retriever for high-speed lexical search.
"""

import logging
import hashlib
from typing import List, Optional
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnableLambda

logger = logging.getLogger(__name__)

class DocumentSearch:
    def __init__(
        self, 
        vector_store: VectorStore, 
        bm25_retriever: Optional[BM25Retriever] = None, 
        strategy: str = "hybrid", 
        top_k: int = 15
    ):
        """
        Initializes the search engine with a provider-agnostic vector store 
        and an optional lexical retriever.
        
        Args:
            vector_store: Any LangChain VectorStore (Chroma, Pinecone, etc.).
            bm25_retriever: A pre-built BM25 retriever cached by the Engine.
            strategy: 'hybrid' or 'semantic'.
            top_k: Number of documents to retrieve in the broad pool.
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.strategy = strategy.lower()
        self.top_k = top_k
        self.retriever = self._initialize_strategy()

    def _build_semantic_retriever(self):
        """Standard vector-based retrieval."""
        return self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

    def _build_hybrid_retriever(self):
        """
        Combines Semantic and Lexical results using Reciprocal Rank Fusion (RRF).
        If no BM25 retriever is available, it gracefully falls back to Semantic.
        """
        semantic_retriever = self._build_semantic_retriever()
        
        if not self.bm25_retriever:
            logger.warning("Hybrid search requested but BM25 index is missing. Falling back to Semantic.")
            return semantic_retriever

        # Synchronize top_k for both retrieval branches
        self.bm25_retriever.k = self.top_k

        # 1. Run retrievers in parallel
        parallel_retrieval = RunnableParallel(
            semantic=semantic_retriever,
            bm25=self.bm25_retriever
        )

        # 2. Custom Reciprocal Rank Fusion (RRF) Logic
        def reciprocal_rank_fusion(results: dict) -> List[Document]:
            """
            Deduplicates and re-scores documents using RRF from both retrieval streams.
            """
            fused_scores = {}
            # Weights favor semantic meaning slightly over exact keyword matches
            weights = {"semantic": 0.7, "bm25": 0.3}
            k_constant = 60 # Standard RRF smoothing constant

            for strategy_name, docs in results.items():
                weight = weights[strategy_name]
                for rank, doc in enumerate(docs, start=1):
                    
                    # Use page_content as a unique ID to deduplicate chunks
                    doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                    
                    if doc_id not in fused_scores:
                        fused_scores[doc_id] = {"doc": doc, "score": 0.0}
                    
                    # RRF Formula: weight * (1 / (k + rank))
                    fused_scores[doc_id]["score"] += weight * (1.0 / (k_constant + rank))

            # Sort by fused score descending
            reranked_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
            
            # Return top_k unique documents
            return [item["doc"] for item in reranked_docs[:self.top_k]]

        # Pipe the components using LCEL
        return parallel_retrieval | RunnableLambda(reciprocal_rank_fusion)

    def _initialize_strategy(self):
        """Selects the retrieval chain based on the chosen strategy."""
        if self.strategy == "semantic": 
            return self._build_semantic_retriever()
        elif self.strategy == "bm25" and self.bm25_retriever: 
            return self.bm25_retriever()
        else: 
            return self._build_hybrid_retriever()

    def search(self, query: str) -> List[Document]:
        """Invokes the retrieval pipeline."""
        return self.retriever.invoke(query)