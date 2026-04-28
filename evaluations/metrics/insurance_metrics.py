"""
insurance_metrics.py
--------------------
Defines custom GEval rubrics specific to the insurance domain.
Uses the globally configured judge and settings to evaluate strict logic adherence.
"""
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from evaluations.eval_config import eval_settings
from evaluations.utils.custom_judge import get_eval_judge

def get_reasoning_metric() -> GEval:
    """
    A custom G-Eval metric that checks if the LLM followed the expected 
    logical reasoning path and included the mandatory policy keywords.
    """
    # Fetch the global judge
    judge = get_eval_judge()
    
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
        - Does the actual output explicitly mention the specific waiting periods, limits, or exclusion codes provided in the expected output?
        - Check specifically for the mandatory keywords listed in the expected output.
        
        Provide your reasoning step-by-step before assigning a score between 0.0 and 1.0.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT, 
            LLMTestCaseParams.ACTUAL_OUTPUT, 
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        model=judge,
        threshold=eval_settings.reasoning_threshold, # Dynamic threshold from config
        strict_mode=True # Forces binary 1 or 0 output for strict compliance
    )