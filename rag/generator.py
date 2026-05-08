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
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.utils import format_retrieved_context
from rag.llm_schemas import PolicyDecision, ComparisonResult

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Handles two-pass prompt population and LLM generation using Google Gemini.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        logger.info(f"Initializing LLM: {model_name} for Stateful Two-Pass Generation")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,  # Deterministic Pass 1 is mandatory
            api_key=api_key,
        )

        # 2. Create strictly bound adjudicators for Pass 1
        self.adjudicator_single = self.llm.with_structured_output(PolicyDecision)
        self.adjudicator_compare = self.llm.with_structured_output(ComparisonResult)

    def generate_single_answer(
        self,
        query: str,
        docs: List[Document],
        policy_name: str,
        history: Optional[List] = None,
    ) -> str:
        """Runs the two-pass pipeline for a single policy with history."""
        if not docs:
            return (
                "No relevant context was found in the policy database for this query."
            )

        context_string = format_retrieved_context(docs, policy_name)

        # --- PASS 1: The Adjudicator (Deterministic JSON) ---
        from rag.prompts import single_policy_decision_template

        decision_chain = single_policy_decision_template | self.adjudicator_single

        decision_response = decision_chain.invoke(
            {"context": context_string, "query": query, "history": history or []}
        )

        clean_json = decision_response.model_dump_json(indent=2)
        logger.info(
            f"=== PASS 1 ADJUDICATOR JSON ===\n{clean_json}\n=============================="
        )

        # --- PASS 2: The Explainer (Markdown UI) ---
        from rag.prompts import single_policy_explainer_template

        explainer_chain = (
            single_policy_explainer_template | self.llm | StrOutputParser()
        )
        final_response = explainer_chain.invoke(
            {"decision_json": clean_json, "query": query}
        )

        return final_response

    def generate_comparison(
        self,
        query: str,
        docs_a: List[Document],
        name_a: str,
        docs_b: List[Document],
        name_b: str,
        history: Optional[List] = None,
    ) -> str:
        """Runs the comparison pipeline across two policies with history."""
        context_a = (
            format_retrieved_context(docs_a, name_a) if docs_a else "No context found."
        )
        context_b = (
            format_retrieved_context(docs_b, name_b) if docs_b else "No context found."
        )

        from rag.prompts import (
            compare_policies_decision_template,
            compare_policies_explainer_template,
        )

        # PASS 1: Adjudicator
        decision_chain = compare_policies_decision_template | self.adjudicator_compare
        decision_response = decision_chain.invoke(
            {
                "context_a": context_a,
                "context_b": context_b,
                "query": query,
                "history": history or [],
            }
        )

        clean_json = decision_response.model_dump_json(indent=2)

        # PASS 2: Explainer
        explainer_chain = (
            compare_policies_explainer_template | self.llm | StrOutputParser()
        )
        final_response = explainer_chain.invoke(
            {"decision_json": clean_json, "query": query}
        )

        return final_response
