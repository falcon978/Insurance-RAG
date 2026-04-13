"""
rag/generator.py
----------------
Manages the LLM and implements the Two-Pass Architecture (Decision -> Explanation).
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.prompts import (
    single_policy_decision_template, 
    single_policy_explainer_template,
    compare_policies_decision_template, 
    compare_policies_explainer_template
)
from rag.utils import format_retrieved_context

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Handles two-pass prompt population and LLM generation using Google Gemini.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        logger.info(f"Initializing LLM: {model_name} for Two-Pass Generation")
        
        # TEMPERATURE 0.0 IS CRITICAL FOR PASS 1 JSON DETERMINISM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )

    def _clean_json_output(self, raw_output: str) -> str:
        """Strips markdown code blocks from the LLM output to ensure clean JSON."""
        return raw_output.replace("```json", "").replace("```", "").strip()

    def generate_single_answer(self, query: str, docs: List[Document], policy_name: str) -> str:
        """Runs the two-pass pipeline for a single policy."""
        if not docs:
            return "No relevant context was found in the policy database for this query."
            
        context_string = format_retrieved_context(docs, policy_name)
        
        # --- PASS 1: The Adjudicator (Deterministic JSON) ---
        decision_chain = single_policy_decision_template | self.llm
        decision_response = decision_chain.invoke({
            "context": context_string, 
            "query": query
        })
        
        clean_json = self._clean_json_output(decision_response.content)
        logger.info(f"\n=== PASS 1: ADJUDICATOR JSON ({policy_name}) ===\n{clean_json}\n=======================================")
        
        # --- PASS 2: The Explainer (Markdown UI) ---
        explainer_chain = single_policy_explainer_template | self.llm
        final_response = explainer_chain.invoke({
            "decision_json": clean_json, 
            "query": query
        })
        
        return final_response.content

    def generate_comparison(self, query: str, docs_a: List[Document], policy_name_a: str, docs_b: List[Document], policy_name_b: str) -> str:
        """Runs the two-pass pipeline across two policies."""
        context_a_string = format_retrieved_context(docs_a, policy_name_a) if docs_a else "No relevant context found."
        context_b_string = format_retrieved_context(docs_b, policy_name_b) if docs_b else "No relevant context found."
        
        # --- PASS 1: The Adjudicator (Deterministic JSON) ---
        decision_chain = compare_policies_decision_template | self.llm
        decision_response = decision_chain.invoke({
            "context_a": context_a_string, 
            "context_b": context_b_string, 
            "query": query
        })
        
        clean_json = self._clean_json_output(decision_response.content)
        logger.info(f"\n=== PASS 1: COMPARISON ADJUDICATOR JSON ===\n{clean_json}\n===========================================")
        
        # --- PASS 2: The Explainer (Markdown UI) ---
        explainer_chain = compare_policies_explainer_template | self.llm
        final_response = explainer_chain.invoke({
            "decision_json": clean_json, 
            "query": query
        })
        
        return final_response.content