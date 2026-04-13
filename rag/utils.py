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
        page_start = doc.metadata.get("page_start", "?")
        page_end = doc.metadata.get("page_end", page_start)
        
        # Display "p. 12" if single page, or "p. 12-13" if spanning
        page_label = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
        
        formatted_text += f"\n\n--- [Source: {policy_name}, Page {page_label}] ---\n"
        formatted_text += f"{doc.page_content}\n"
    return formatted_text