"""
rag/generator.py
----------------
Manages the LLM and implements the Unified Single-Pass Architecture (Chain of Thought).
Uses a custom LangChain Output Parser (SentinelOutputParser) to safely extract
JSON and Markdown from unique sentinels.
"""

import re
import json
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.utils import format_retrieved_context
from rag.generation_prompts import (
    single_policy_unified_template,
    compare_policies_unified_template,
)

logger = logging.getLogger(__name__)


class SentinelOutputParser(BaseOutputParser[dict]):
    """
    A custom LangChain parser that extracts JSON and Markdown
    fenced by strict <<<BEGIN>>> and <<<END>>> sentinels.
    """

    def parse(self, text: str) -> dict:
        try:
            # 1. Extract and parse Adjudicator JSON
            json_match = re.search(
                r"<<<BEGIN_ADJUDICATION_JSON>>>(.*?)<<<END_ADJUDICATION_JSON>>>",
                text,
                re.DOTALL,
            )
            adjudicator_str = json_match.group(1).strip() if json_match else "{}"

            try:
                adjudicator_json = json.loads(adjudicator_str)
                logger.info(
                    f"=== INTERNAL ADJUDICATOR JSON ===\n{json.dumps(adjudicator_json, indent=2)}\n================================="
                )
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse Adjudicator JSON. Raw string: {adjudicator_str}"
                )
                adjudicator_json = {"error": "Malformed JSON output from LLM."}

            # 2. Extract the Advisory Report Markdown
            report_match = re.search(
                r"<<<BEGIN_ADVISORY_REPORT>>>(.*?)<<<END_ADVISORY_REPORT>>>",
                text,
                re.DOTALL,
            )
            advisory_report = report_match.group(1).strip() if report_match else None

            if not advisory_report:
                logger.error(
                    f"Failed to find <<<BEGIN_ADVISORY_REPORT>>> sentinels.\nRaw Output: {text}"
                )
                advisory_report = "An error occurred while generating the advisory report. Please try again."

            # Clean up markdown formatting artifacts if the LLM adds them inside the tags
            if advisory_report.startswith("```markdown"):
                advisory_report = advisory_report[11:]
            if advisory_report.endswith("```"):
                advisory_report = advisory_report[:-3]

            return {
                "adjudication": adjudicator_json,
                "advisory_report": advisory_report.strip(),
            }

        except Exception as e:
            logger.error(f"Fatal parsing error: {e}\nRaw Output: {text}")
            return {
                "adjudication": {"error": str(e)},
                "advisory_report": "An internal parsing error occurred.",
            }


class ResponseGenerator:
    """
    Handles prompt population and LLM generation.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        logger.info(f"Initializing generator with model: {model_name}")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            api_key=api_key,
        )
        # Instantiate the LCEL parser once
        self.parser = SentinelOutputParser()

    async def a_generate_single_answer(
        self,
        query: str,
        docs: List[Document],
        policy_name: str,
        history: Optional[List] = None,
    ) -> str:
        """Executes the generation pipeline for a single policy asynchronously with history."""
        if not docs:
            return (
                "No relevant context was found in the policy database for this query."
            )

        context_string = format_retrieved_context(docs, policy_name)

        # LCEL execution: Prompt -> LLM -> Custom Sentinel Parser
        chain = single_policy_unified_template | self.llm | self.parser

        # ainvoke() returns the dictionary defined in SentinelOutputParser
        parsed_result = await chain.ainvoke(
            {"context": context_string, "query": query, "history": history or []}
        )

        # Return just the Markdown report for the UI
        return parsed_result["advisory_report"]

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
        context_a = (
            format_retrieved_context(docs_a, name_a) if docs_a else "No context found."
        )
        context_b = (
            format_retrieved_context(docs_b, name_b) if docs_b else "No context found."
        )

        # LCEL execution: Prompt -> LLM -> Custom Sentinel Parser
        chain = compare_policies_unified_template | self.llm | self.parser

        # ainvoke() returns the dictionary defined in SentinelOutputParser
        parsed_result = await chain.ainvoke(
            {
                "context_a": context_a,
                "context_b": context_b,
                "query": query,
                "history": history or [],
            }
        )

        # Return just the Markdown report for the UI
        return parsed_result["advisory_report"]
