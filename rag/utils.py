"""
rag/utils.py
------------
Utility functions for text formatting and metadata extraction.
"""

from typing import List
from langchain_core.documents import Document

def format_retrieved_context(docs: List[Document], policy_name: str) -> str:
    """
    Takes a list of retrieved LangChain documents and formats them into a single 
    string, safely injecting the hidden page metadata back into the readable text.
    
    Args:
        docs: List of documents retrieved from ChromaDB/BM25.
        policy_name: The human-readable name of the policy collection.
    """
    formatted_text = ""
    for doc in docs:
        # Default to 'Unknown' if the metadata was somehow lost during chunking
        page = doc.metadata.get("page_start", "Unknown")
        formatted_text += f"\n\n--- [Source: {policy_name}, Page {page}] ---\n{doc.page_content}"
        
    return formatted_text