"""
rag/generator.py
----------------
Manages the LLM and implements the Two-Pass Architecture (Decision -> Explanation).
Hardened with regex JSON extraction and support for chat history injection.
"""

import re
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.utils import format_retrieved_context

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Handles two-pass prompt population and LLM generation using Google Gemini.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        logger.info(f"Initializing LLM: {model_name} for Stateful Two-Pass Generation")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, # Deterministic Pass 1 is mandatory
            api_key=api_key
        )

    def _extract_json(self, raw_output) -> str:
        """
        Uses regex to isolate the JSON object.
        Prevents failures if the LLM outputs markdown blocks, preamble text,
        or structured list blocks from newer LangChain versions.
        """
        # --- NEW FIX: Handle LangChain list outputs ---
        if isinstance(raw_output, list):
            # Extract text from dict blocks (or cast to string if plain items)
            raw_output = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block) 
                for block in raw_output
            )
        elif not isinstance(raw_output, str):
            raw_output = str(raw_output)
        # ----------------------------------------------

        match = re.search(r'(\{.*\}|\[.*\])', raw_output, re.DOTALL)
        if match:
            return match.group(1)
            
        # Fallback: strip markdown if regex fails but structure exists
        return raw_output.replace("```json", "").replace("```", "").strip()

    def generate_single_answer(self, query: str, docs: List[Document], policy_name: str, history: Optional[List] = None) -> str:
        """Runs the two-pass pipeline for a single policy with history."""
        if not docs:
            return "No relevant context was found in the policy database for this query."
            
        context_string = format_retrieved_context(docs, policy_name)
        
        # --- PASS 1: The Adjudicator (Deterministic JSON) ---
        from rag.prompts import single_policy_decision_template
        decision_chain = single_policy_decision_template | self.llm
        
        decision_response = decision_chain.invoke({
            "context": context_string, 
            "query": query,
            "history": history or []
        })
        
        clean_json = self._extract_json(decision_response.content)
        logger.info(f"=== PASS 1 ADJUDICATOR JSON ===\n{clean_json}\n==============================")
        
        # --- PASS 2: The Explainer (Markdown UI) ---
        from rag.prompts import single_policy_explainer_template
        explainer_chain = single_policy_explainer_template | self.llm
        final_response = explainer_chain.invoke({
            "decision_json": clean_json, 
            "query": query
        })
        
        return final_response.content

    def generate_comparison(self, query: str, docs_a: List[Document], name_a: str, docs_b: List[Document], name_b: str, history: Optional[List] = None) -> str:
        """Runs the comparison pipeline across two policies with history."""
        context_a = format_retrieved_context(docs_a, name_a) if docs_a else "No context found."
        context_b = format_retrieved_context(docs_b, name_b) if docs_b else "No context found."
        
        from rag.prompts import compare_policies_decision_template, compare_policies_explainer_template
        
        # PASS 1: Adjudicator
        decision_chain = compare_policies_decision_template | self.llm
        decision_response = decision_chain.invoke({
            "context_a": context_a, 
            "context_b": context_b, 
            "query": query,
            "history": history or []
        })
        
        clean_json = self._extract_json(decision_response.content)
        
        # PASS 2: Explainer
        explainer_chain = compare_policies_explainer_template | self.llm
        final_response = explainer_chain.invoke({
            "decision_json": clean_json, 
            "query": query
        })
        
        return final_response.content