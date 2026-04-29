"""
test_standard_rag_metrics.py
----------------------------
Executes the core RAG Triad (Recall, Precision, Faithfulness, Relevancy) + Custom Logic Metric
across the entire golden dataset.
"""
import pytest
import time
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric
)

from evaluations.utils.rag_wrapper import EvalRAGWrapper
from evaluations.datasets.data_loader import load_golden_dataset
from evaluations.metrics.insurance_metrics import get_reasoning_metric
from evaluations.eval_config import eval_settings
from evaluations.utils.custom_judge import get_eval_judge

# 1. Initialize Wrapper and Global Judge globally 
# (This is safe because they do not hold test-specific state, and we don't want 
# to reload the embedding model / LLM connections on every single loop)
rag = EvalRAGWrapper()
eval_judge = get_eval_judge()

@pytest.mark.parametrize(
    "case_id, query, source, expected_snippets, keywords, reasoning", 
    load_golden_dataset()
)
def test_full_rag_triad(case_id, query, source, expected_snippets, keywords, reasoning):
    
    # Outer pause to space out the start of each test case
    time.sleep(eval_settings.rate_limit_delay_seconds)
    
    # Standard Execution
    actual_output, retrieved_contexts = rag.query(
        query=query, 
        source=source, # The wrapper handles the mapping!
        retrieve_top_k=eval_settings.retrieve_top_k, 
        rerank_top_k=eval_settings.rerank_top_k
    )
    
    expected_output_str = f"Keywords to include: {', '.join(keywords)}. Logic: {reasoning}"
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output_str,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets
    )
    
    # 2. Initialize standard metrics LOCALLY inside the test function.
    # This guarantees that each test case evaluates using a fresh metric instance,
    # preventing the internal scores and reasons from leaking across loops.
    recall_metric = ContextualRecallMetric(threshold=eval_settings.recall_threshold, model=eval_judge, include_reason=True)
    precision_metric = ContextualPrecisionMetric(threshold=eval_settings.precision_threshold, model=eval_judge, include_reason=True)
    faithfulness_metric = FaithfulnessMetric(threshold=eval_settings.faithfulness_threshold, model=eval_judge, include_reason=True)
    relevancy_metric = AnswerRelevancyMetric(threshold=eval_settings.relevancy_threshold, model=eval_judge, include_reason=True)
    
    # Ensure the custom factory also returns a fresh instance
    reasoning_metric = get_reasoning_metric()
    
    # 3. Assert test sequentially to respect API rate limits
    assert_test(
        test_case, 
        [
            recall_metric,       # Evaluates embedding model (Recall@K)
            precision_metric,    # Evaluates reranker/ranking (MRR)
            faithfulness_metric, # Evaluates LLM hallucination
            relevancy_metric,    # Evaluates generic answer quality
            reasoning_metric     # Evaluates strict clause logic
        ], 
        run_async=False # Forces metrics to evaluate one-by-one
    )