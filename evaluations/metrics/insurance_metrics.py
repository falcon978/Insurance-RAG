# evaluations/metrics/insurance_metrics.py
import os
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# Configurable Judge Model - defaults to OpenAI, but can be set to Gemini
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")

def get_reasoning_metric(threshold: float = 0.7) -> GEval:
    """
    A custom G-Eval metric that checks if the LLM followed the expected 
    logical reasoning path and included the mandatory keywords.
    """
    return GEval(
        name="Policy Reasoning Adherence",
        criteria="""
        You are an expert insurance auditor evaluating an AI's response to a policyholder's query.
        
        Evaluate the actual output against the expected output using the following strict rubric:
        
        Step 1: The Ultimate Outcome (Pass/Fail)
        - Does the actual output reach the exact same final conclusion (e.g., "Covered", "Not Covered", "Conditional") as the expected logic?
        - IF NO: The score must be 0. Do not proceed to Step 2.
        
        Step 2: The Reasoning Path
        - Does the actual output follow the logical steps outlined in the expected logic?
        - IF NO: Deduct points heavily. The AI cannot arrive at the right answer using the wrong policy logic.
        
        Step 3: Policy Constraints & Keywords
        - Does the actual output explicitly mention the specific waiting periods, limits, or exclusion codes (e.g., "24 months", "Excl02") provided in the expected output?
        - Check specifically for the mandatory keywords listed in the expected output.
        
        Provide your reasoning step-by-step before assigning a score between 0.0 and 1.0.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT, 
            LLMTestCaseParams.ACTUAL_OUTPUT, 
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        model=JUDGE_MODEL,
        threshold=threshold,
        strict_mode=True # Forces binary 1 or 0 output for strict compliance
    )