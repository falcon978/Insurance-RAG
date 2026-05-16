"""
rag/rerankers.py
----------------
Houses the cross-encoder logic for scoring and re-ordering retrieved documents.
Isolating this allows for easy swapping of reranking strategies (e.g., Cohere API, BGE-Reranker)
without altering the core retrieval logic.
"""

import logging
import asyncio
from typing import List, Tuple, Union
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class ContextReranker:
    """
    Uses a HuggingFace Cross-Encoder to rerank documents based on exact query-to-chunk context.
    Unlike Bi-Encoders (which process queries and docs separately), Cross-Encoders process
    them simultaneously for highly accurate semantic matching.
    """

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"
    ):
        logger.info(f"Loading Cross-Encoder model: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4,
        return_scores: bool = False,
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Scores document relevance and returns the top K highest-scoring chunks.

        Args:
            query: The user's question.
            documents: The broad list of candidate LangChain Document objects.
            top_k: The final number of documents to retain.
            return_scores: If True, returns a list of (Document, score) tuples.

        Returns:
            A sorted list of the top_k Document objects (or tuples if return_scores=True).
        """
        if not documents:
            return []

        # CrossEncoder expects a list of [query, text] pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Predict returns a numpy array of logits/scores
        scores = self.model.predict(pairs)

        # Zip the original Document objects with their new scores
        scored_docs = list(zip(documents, scores))

        # Sort descending (highest score first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if return_scores:
            # Notebooks and diagnostics use this to plot confidence distributions
            return [(doc, float(score)) for doc, score in scored_docs[:top_k]]

        # Extract and return just the Document objects for the standard pipeline
        return [doc for doc, score in scored_docs[:top_k]]

    async def a_rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4,
        return_scores: bool = False,
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Safely executes the CPU-bound reranking task in a background thread."""
        return await asyncio.to_thread(
            self.rerank, query, documents, top_k, return_scores
        )
