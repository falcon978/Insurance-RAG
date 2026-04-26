# evaluations/test_cases/test_standard_rag_metrics.py
import pytest
import os
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

# Initialize Wrapper and Configs
rag = EvalRAGWrapper()
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")

# Initialize standard metrics globally for the suite
recall_metric = ContextualRecallMetric(threshold=0.8, model=JUDGE_MODEL, include_reason=True)
precision_metric = ContextualPrecisionMetric(threshold=0.8, model=JUDGE_MODEL, include_reason=True)
faithfulness_metric = FaithfulnessMetric(threshold=0.9, model=JUDGE_MODEL, include_reason=True)
relevancy_metric = AnswerRelevancyMetric(threshold=0.8, model=JUDGE_MODEL, include_reason=True)
reasoning_metric = get_reasoning_metric()

@pytest.mark.parametrize(
    "case_id, query, source, expected_snippets, keywords, reasoning", 
    load_golden_dataset()
)
def test_full_rag_triad(case_id, query, source, expected_snippets, keywords, reasoning):
    collection = "hdfc_care_docs" if source == "both" else f"{source}_docs"
    
    # Standard Execution
    actual_output, retrieved_contexts = rag.query(
        query=query, 
        collection_name=collection, 
        retrieve_top_k=10, 
        rerank_top_k=3
    )
    
    expected_output_str = f"Keywords to include: {', '.join(keywords)}. Logic: {reasoning}"
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output_str,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets
    )
    
    assert_test(
        test_case, 
        [
            recall_metric,       # Evaluates embedding model (Recall@K)
            precision_metric,    # Evaluates reranker/ranking (MRR)
            faithfulness_metric, # Evaluates LLM hallucination
            relevancy_metric,    # Evaluates generic answer quality
            reasoning_metric     # Evaluates strict clause logic
        ]
    )