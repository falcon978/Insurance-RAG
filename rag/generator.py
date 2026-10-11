"""
rag/generator.py
----------------
Manages the LLM and implements the Unified Single-Pass Architecture (Chain of Thought).
Uses Langchain's native with_structured_output to enforce strict Pydantic JSON schemas.
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from rag.utils import format_retrieved_context, render_advisory_markdown
from rag.generation_prompts import (
    single_policy_unified_template,
    compare_policies_unified_template,
)
from rag.llm_schemas import SinglePolicyResponse, ComparePolicyResponse
from rag.exceptions import LLMGenerationError

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Handles prompt population and LLM generation using native structured outputs.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def a_generate_single_answer(
        self,
        query: str,
        docs: List[Document],
        policy_name: str,
        history: Optional[List] = None,
    ) -> str:
        """Executes the generation pipeline for a single policy asynchronously with history."""
        if not docs:
            return "No relevant context was found in the policy database for this query."

        context_string = format_retrieved_context(docs, policy_name)

        formatted_history = []
        for msg in history or []:
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg.get("content", "")))

        # Create structured LLM chain
        structured_llm = self.llm.with_structured_output(SinglePolicyResponse)
        chain = single_policy_unified_template | structured_llm

        try:
            parsed_result = await chain.ainvoke(
                {"context": context_string, "query": query, "history": formatted_history}
            )
            # The result is a Pydantic object. Dump it to a dict for the renderer.
            report_payload = parsed_result.advisory_report.model_dump()
        except Exception as e:
            logger.error(f"Structured output parsing failed: {e}")
            raise LLMGenerationError(f"Failed to generate structured response: {e}")

        # Route the dictionary through the markdown formatter
        return render_advisory_markdown(report_payload)

    async def a_generate_comparison(
        self,
        query: str,
        docs_a: List[Document],
        name_a: str,
        docs_b: List[Document],
        name_b: str,
        history: Optional[List] = None,
    ) -> str:
        """Executes the comparative generation pipeline across two policies asynchronously."""
        context_a = format_retrieved_context(docs_a, name_a) if docs_a else "No context found."
        context_b = format_retrieved_context(docs_b, name_b) if docs_b else "No context found."

        formatted_history = []
        for msg in history or []:
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg.get("content", "")))

        # Create structured LLM chain
        structured_llm = self.llm.with_structured_output(ComparePolicyResponse)
        chain = compare_policies_unified_template | structured_llm

        try:
            parsed_result = await chain.ainvoke(
                {
                    "context_a": context_a,
                    "context_b": context_b,
                    "query": query,
                    "history": formatted_history,
                }
            )
            report_payload = parsed_result.advisory_report.model_dump()
        except Exception as e:
            logger.error(f"Structured output parsing failed: {e}")
            raise LLMGenerationError(f"Failed to generate structured comparison: {e}")

        return render_advisory_markdown(report_payload)
