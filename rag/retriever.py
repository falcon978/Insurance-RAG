"""
rag/retriever.py
----------------
Handles document retrieval (Semantic, Lexical, Hybrid) and Cross-Encoder Reranking.
"""

import logging
from typing import List
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class DocumentSearch:
    """
    Wraps the Chroma DB to provide multiple search strategies.
    Supports standard vector similarity (Semantic), keyword search (BM25), 
    and Reciprocal Rank Fusion (Hybrid).
    """
    def __init__(self, vector_store: Chroma, strategy: str = "hybrid", top_k: int = 15):
        self.vector_store = vector_store
        self.strategy = strategy.lower()
        self.top_k = top_k
        self.retriever = self._initialize_strategy()

    def _build_semantic_retriever(self):
        """Builds a standard vector-based retriever."""
        logger.info(f"Initializing Semantic Retriever (Top K: {self.top_k})")
        return self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

    def _build_bm25_retriever(self):
        """Extracts text from Chroma to build an in-memory lexical BM25 index."""
        logger.info(f"Initializing BM25 Retriever (Top K: {self.top_k})...")
        
        db_data = self.vector_store.get()
        docs = [
            Document(page_content=txt, metadata=meta) 
            for txt, meta in zip(db_data['documents'], db_data['metadatas'])
        ]
        
        if not docs:
            logger.warning("Database is empty. Cannot build BM25.")
            return None
            
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = self.top_k
        return bm25_retriever

    def _build_hybrid_retriever(self):
        """Combines Semantic and BM25 retrievers using Reciprocal Rank Fusion."""
        logger.info("Initializing Hybrid Ensemble Retriever (RRF)...")
        
        semantic_retriever = self._build_semantic_retriever()
        bm25_retriever = self._build_bm25_retriever()
        
        # Safe fallback if the database is empty or BM25 fails
        if not bm25_retriever:
            logger.warning("Falling back to pure Semantic Search.")
            return semantic_retriever
            
        return EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.7, 0.3] # Prioritize meaning, but respect exact keyword matches
        )

    def _initialize_strategy(self):
        """Routes to the correct builder based on the requested strategy."""
        if self.strategy == "semantic":
            return self._build_semantic_retriever()
        elif self.strategy == "bm25":
            # If BM25 fails (e.g. empty DB), gracefully fall back to semantic
            return self._build_bm25_retriever() or self._build_semantic_retriever()
        else:
            return self._build_hybrid_retriever()

    def search(self, query: str) -> List[Document]:
        """Executes the search and returns the broad candidate documents."""
        return self.retriever.invoke(query)


class ContextReranker:
    """
    Uses a HuggingFace Cross-Encoder to rerank documents based on exact query-to-chunk context.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', device: str = "cpu"):
        logger.info(f"Loading Cross-Encoder model: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)
        
    def rerank(self, query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
        """Scores document relevance and returns the top K highest-scoring chunks."""
        if not documents:
            return []
            
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]