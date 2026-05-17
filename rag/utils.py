"""
rag/utils.py
------------
Utility functions for text formatting, metadata extraction, and mathematical fusion.
"""

from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


def format_retrieved_context(docs: List[Document], policy_name: str) -> str:
    """
    Takes a list of retrieved LangChain documents and formats them into a single
    string, safely injecting the hidden page metadata back into the readable text.

    Args:
        docs: List of documents retrieved from ChromaDB/BM25.
        policy_name: The human-readable name of the policy collection.

    Returns:
        A formatted markdown string ready to be injected into the LLM context window.
    """
    formatted_text = ""
    for doc in docs:
        page_start = doc.metadata.get("page_start", "?")
        page_end = doc.metadata.get("page_end", page_start)

        page_label = (
            f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
        )

        formatted_text += f"\n\n--- [Source: {policy_name}, Page {page_label}] ---\n"
        formatted_text += f"{doc.page_content}\n"
    return formatted_text


def fuse_multi_query_results(
    results_list: List[List[Document]], k: int = 60
) -> List[Document]:
    """
    Applies Standard Reciprocal Rank Fusion (RRF) across an arbitrary number of document lists.
    Used for pooling results from similar retrieval tracks (e.g., multiple vector searches)
    where all tracks are weighted equally.

    Args:
        results_list: A list of lists containing LangChain Document objects.
        k: A smoothing constant used in the RRF formula (default 60 is industry standard).

    Returns:
        A single, deduplicated, and ranked List of LangChain Documents.
    """
    fused_scores = {}
    doc_map = {}

    for doc_list in results_list:
        # Enumerate gives us the 'rank' (index) of the document in its specific list
        for rank, doc in enumerate(doc_list):

            # Use the actual text content as a unique hash identifier.
            # This prevents duplicating the same insurance clause if it was retrieved
            # independently by both BM25 and Pinecone/Chroma.
            doc_hash = hash(doc.page_content)

            if doc_hash not in fused_scores:
                fused_scores[doc_hash] = 0
                doc_map[doc_hash] = doc

            # RRF Formula: 1 / (k + rank)
            # The higher the original rank (lower index), the larger the added score.
            fused_scores[doc_hash] += 1 / (rank + k)

    # Sort the hashes based on their accumulated RRF score in descending order
    reranked_hashes = sorted(
        fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True
    )

    # Map the hashes back to their original Document objects in the new order
    return [doc_map[doc_hash] for doc_hash in reranked_hashes]


def fuse_weighted_results(
    list_a: List[Document],
    list_b: List[Document],
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    k: int = 60,
) -> List[Document]:
    """
    Applies Weighted Reciprocal Rank Fusion.

    Used for balancing orthogonal retrieval tracks (e.g., Semantic Vector vs. Lexical BM25).
    Allows explicit mathematical control over the influence of each retrieval method.

    Args:
        list_a: First list of LangChain Documents.
        list_b: Second list of LangChain Documents.
        weight_a: Multiplier for the RRF score of list_a (default 0.5).
        weight_b: Multiplier for the RRF score of list_b (default 0.5).
        k: Smoothing constant.

    Returns:
        List[Document]: A single, deduplicated, and ranked list.
    """
    fused_scores = {}
    doc_map = {}

    # Process List A
    for rank, doc in enumerate(list_a):
        doc_hash = hash(doc.page_content)
        if doc_hash not in fused_scores:
            fused_scores[doc_hash] = 0
            doc_map[doc_hash] = doc
        fused_scores[doc_hash] += weight_a * (1 / (rank + k))

    # Process List B
    for rank, doc in enumerate(list_b):
        doc_hash = hash(doc.page_content)
        if doc_hash not in fused_scores:
            fused_scores[doc_hash] = 0
            doc_map[doc_hash] = doc
        fused_scores[doc_hash] += weight_b * (1 / (rank + k))

    reranked_hashes = sorted(
        fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True
    )
    return [doc_map[doc_hash] for doc_hash in reranked_hashes]


def render_advisory_markdown(response_data: dict) -> str:
    """
    Converts the structured JSON output from the Sentinel Output Parser
    into readable Markdown.
    """
    if isinstance(response_data, str):
        return response_data

    md = ""

    if "executive_summary" in response_data:
        md += f"### 📑 Executive Summary\n{response_data['executive_summary']}\n\n"

    if "comparison_table" in response_data:
        md += f"### ⚖️ Policy Comparison\n{response_data['comparison_table']}\n\n"

    if "deep_dive" in response_data:
        md += f"### 🔍 Deep Dive\n{response_data['deep_dive']}\n\n"

    if "risk_disclosure" in response_data:
        md += f"### ⚠️ Risk Disclosure\n{response_data['risk_disclosure']}\n\n"

    if "follow_up_questions" in response_data and response_data["follow_up_questions"]:
        md += "### 📋 Action Items for Insurer / Client\n"
        md += "*To finalize this adjudication, please obtain the following:* \n"
        for q in response_data["follow_up_questions"]:
            md += f"- {q}\n"
        md += "\n"

    if "mandatory_disclaimer" in response_data:
        md += "---\n"
        md += f"*{response_data['mandatory_disclaimer']}*"

    return md
