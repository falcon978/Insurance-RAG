"""
insurance_metrics.py
--------------------

This module defines custom GEval metrics using DeepEval to evaluate the
Insurance RAG pipeline. It intentionally decouples 'Answer Correctness'
from 'Reasoning Faithfulness' to prevent false negatives caused by LLMs
phrasing correct answers differently than the expected golden datasets.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from evaluations.utils.custom_judge import get_eval_judge


def get_answer_correctness_metric() -> GEval:
    """
    Creates a GEval metric to evaluate the semantic correctness of the final answer.

    This metric focuses exclusively on the "What" (the ultimate outcome/decision).
    It is lenient regarding phrasing and structure, allowing the LLM to pass as long
    as its final conclusion (e.g., Covered, Not Covered, Conditional) aligns with
    the expected output. It explicitly ignores the reasoning steps.

    Returns:
        GEval: A configured DeepEval metric for Answer Correctness.
    """
    judge = get_eval_judge()

    return GEval(
        name="Answer Correctness [GEval]",
        criteria="""
        You are evaluating whether the model reached the correct conclusion.

        Instructions:
        - Compare ACTUAL OUTPUT with EXPECTED OUTPUT
        - Focus ONLY on the final answer/conclusion

        Rules:
        - Allow paraphrasing
        - Allow additional correct details
        - Allow conditional phrasing if it aligns with expected answer

        Examples:
        - "Covered after 24 months" == "Covered with a 2-year waiting period"
        - "Conditional coverage" is acceptable if it matches expected logic

        Mark as INCORRECT only if:
        - The conclusion contradicts the expected answer
        - OR the conclusion is missing

        Ignore reasoning quality for this evaluation.
        """,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=0.7,
        # strict_mode=False allows for continuous scoring (0.0 to 1.0) instead of
        # a binary 0 or 1, accommodating partial correctness or nuanced answers.
        strict_mode=False,
    )


def get_reasoning_faithfulness_metric() -> GEval:
    """
    Creates a GEval metric to evaluate the logical grounding of the LLM's response.

    This metric focuses exclusively on the "Why" (the reasoning path). It acts as
    a strict insurance auditor, ensuring the LLM reached its conclusion using the
    correct policy clauses, constraints (e.g., waiting periods, specific exclusion
    codes), and explicit text from the retrieved context.

    Returns:
        GEval: A configured DeepEval metric for Reasoning Faithfulness.
    """
    judge = get_eval_judge()

    return GEval(
        name="Reasoning Faithfulness [GEval]",
        criteria="""
        You are an expert insurance auditor evaluating the reasoning quality.

        Evaluate whether the reasoning in ACTUAL OUTPUT is faithful to policy logic based on the EXPECTED OUTPUT.

        Step 1: Policy Alignment
        - Is the reasoning consistent with policy rules?
        - Penalize any incorrect assumptions or invented rules

        Step 2: Constraint Coverage
        - Are key constraints mentioned?
          (e.g., waiting periods, exclusions, limits)

        Step 3: Grounding
        - Is the reasoning clearly grounded in policy text?
        - Penalize vague or generic insurance logic

        Important:
        - The reasoning does NOT need to match the expected output step-by-step
        - Alternative valid reasoning paths are acceptable

        Scoring:
        - 1.0 -> Fully correct and policy-grounded reasoning
        - 0.5 -> Partially correct but missing constraints or slightly vague
        - 0.0 -> Incorrect or hallucinated reasoning
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=0.7,
        # strict_mode=False is crucial here so the LLM can output the 0.5 score
        # defined in the criteria for partially correct/vague reasoning.
        strict_mode=False,
    )
