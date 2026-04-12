"""
rag/generator.py
----------------
Manages the LLM and processes the retrieved contexts into final answers.
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.prompts import single_policy_template, compare_policies_template
from rag.utils import format_retrieved_context

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Handles prompt population and LLM generation using Google Gemini.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        logger.info(f"Initializing LLM: {model_name}")
        # Note: Using gemini-1.5-flash as the stable endpoint for the free tier. 
        # Update to 3.0 or 2.0 based on your specific Google AI Studio access.
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.1, # Low temp for strict, factual extraction
            google_api_key=api_key
        )

    def generate_single_answer(self, query: str, docs: list, policy_name: str) -> str:
        """Generates an answer based on context from a single insurance policy."""
        if not docs:
            return "No relevant context was found in the policy database for this query."
            
        context_string = format_retrieved_context(docs, policy_name)
        
        chain = single_policy_template | self.llm
        response = chain.invoke({
            "context": context_string, 
            "query": query
        })
        
        return response.content

    def generate_comparison(self, query: str, docs_a: list, policy_name_a: str, docs_b: list, policy_name_b: str) -> str:
        """Generates a side-by-side comparison using contexts from two separate policies."""
        context_a_string = format_retrieved_context(docs_a, policy_name_a) if docs_a else "No relevant context found."
        context_b_string = format_retrieved_context(docs_b, policy_name_b) if docs_b else "No relevant context found."
        
        chain = compare_policies_template | self.llm
        response = chain.invoke({
            "context_a": context_a_string, 
            "context_b": context_b_string, 
            "query": query
        })
        
        return response.content