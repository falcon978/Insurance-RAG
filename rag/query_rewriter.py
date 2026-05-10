"""
rag/query_rewriter.py
---------------------
Implements the Translation Layer for Dual-Track Retrieval.
Translates colloquial user queries into formal legal/insurance terminology
to bridge the "Vocabulary Mismatch" gap in Semantic Vector databases.
"""

from rag.llm_schemas import OptimizedSearchQuery
from rag.retrieval_prompts import STRUCTURED_TRANSLATOR_PROMPT


def get_structured_rewriter_chain(fast_llm):
    """
    Creates a LangChain pipeline that translates a user query into a
    structured OptimizedSearchQuery object.

    Args:
        fast_llm: A low-latency LLM instance (e.g., Gemini 1.5 Flash)
                  optimized for quick preprocessing tasks.

    Returns:
        Runnable: A LangChain executable pipeline (Prompt -> LLM -> String).
    """
    # Use LCEL to pipe the imported prompt into the LLM and extract the string
    structured_llm = fast_llm.with_structured_output(OptimizedSearchQuery)

    return STRUCTURED_TRANSLATOR_PROMPT | structured_llm
