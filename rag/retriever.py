"""
rag/retriever.py
----------------
Handles document retrieval (Semantic, Lexical, Hybrid).
Reranking has been decoupled into rerankers.py.
"""

import logging
from typing import List
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnableLambda

logger = logging.getLogger(__name__)

class DocumentSearch:
    def __init__(self, vector_store: Chroma, strategy: str = "hybrid", top_k: int = 15):
        self.vector_store = vector_store
        self.strategy = strategy.lower()
        self.top_k = top_k
        self.retriever = self._initialize_strategy()

    def _build_semantic_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

    def _build_bm25_retriever(self):
        db_data = self.vector_store.get()
        docs = [
            Document(page_content=txt, metadata=meta) 
            for txt, meta in zip(db_data['documents'], db_data['metadatas'])
        ]
        if not docs: 
            return None
            
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = self.top_k
        return bm25_retriever

    def _build_hybrid_retriever(self):
        semantic_retriever = self._build_semantic_retriever()
        bm25_retriever = self._build_bm25_retriever()
        
        if not bm25_retriever:
            return semantic_retriever

        # 1. Run both retrievers in parallel asynchronously
        parallel_retrieval = RunnableParallel(
            semantic=semantic_retriever,
            bm25=bm25_retriever
        )

        # 2. Custom Reciprocal Rank Fusion (RRF) Logic
        def reciprocal_rank_fusion(results: dict) -> List[Document]:
            """
            Takes the results from Semantic and BM25, applies RRF math, 
            and returns a single deduplicated, reranked list of documents.
            """
            fused_scores = {}
            # We apply slight weights to favor semantic meaning over exact keyword matching
            weights = {"semantic": 0.7, "bm25": 0.3}
            k_constant = 60 # Standard constant used in RRF algorithms

            for strategy_name, docs in results.items():
                weight = weights[strategy_name]
                for rank, doc in enumerate(docs, start=1):
                    # Use page_content as a unique ID to deduplicate chunks found by both algorithms
                    doc_id = doc.page_content 
                    
                    if doc_id not in fused_scores:
                        fused_scores[doc_id] = {"doc": doc, "score": 0.0}
                    
                    # RRF Formula: weight * (1 / (k + rank))
                    fused_scores[doc_id]["score"] += weight * (1.0 / (k_constant + rank))

            # Sort the dictionary by the fused score descending
            reranked_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
            
            # Return just the Document objects, sliced to top_k
            return [item["doc"] for item in reranked_docs[:self.top_k]]

        # 3. Pipe them together using LCEL
        hybrid_chain = parallel_retrieval | RunnableLambda(reciprocal_rank_fusion)
        return hybrid_chain

    def _initialize_strategy(self):
        if self.strategy == "semantic": 
            return self._build_semantic_retriever()
        elif self.strategy == "bm25": 
            return self._build_bm25_retriever() or self._build_semantic_retriever()
        else: 
            return self._build_hybrid_retriever()

    def search(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)