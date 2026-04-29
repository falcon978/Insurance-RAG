"""
test_hybrid_vs_semantic.py
--------------------------
A/B testing for Contextual Recall.
Measures the retrieval performance of Pure Vector Search vs. Hybrid Search (Vector + BM25).
"""
import pytest
import time
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

from evaluations.utils.rag_wrapper import EvalRAGWrapper
from evaluations.datasets.data_loader import load_golden_dataset
from evaluations.eval_config import eval_settings
from evaluations.utils.custom_judge import get_eval_judge

rag = EvalRAGWrapper()
eval_judge = get_eval_judge()

# ---------------------------------------------------------
# TEST A: Pure Semantic Baseline
# ---------------------------------------------------------
@pytest.mark.parametrize("case_id, query, source, expected_snippets, keywords, reasoning", load_golden_dataset())
def test_pure_semantic_search(case_id, query, source, expected_snippets, keywords, reasoning):
    
    time.sleep(eval_settings.rate_limit_delay_seconds)
    
    # Isolate vector search by forcing strategy="semantic"
    actual_output, retrieved_contexts = rag.query(
        query=query, 
        source=source, # The wrapper handles the mapping!
        retrieve_top_k=eval_settings.retrieve_top_k, 
        rerank_top_k=eval_settings.rerank_top_k, # Set it to 0 to isolate retrieval performance without the reranker influence
        strategy="semantic"
    )
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets
    )
    
    # CRITICAL FIX: Instantiate metric inside the test case to prevent state leakage
    recall_metric = ContextualRecallMetric(
        threshold=eval_settings.recall_threshold, 
        model=eval_judge, 
        include_reason=True
    )
    
    assert_test(test_case, [recall_metric], run_async=False)

# ---------------------------------------------------------
# TEST B: Hybrid Search (Semantic + BM25)
# ---------------------------------------------------------
@pytest.mark.parametrize("case_id, query, source, expected_snippets, keywords, reasoning", load_golden_dataset())
def test_hybrid_search(case_id, query, source, expected_snippets, keywords, reasoning):
    
    time.sleep(eval_settings.rate_limit_delay_seconds)
    
    # Enable lexical fusion by forcing strategy="hybrid"
    actual_output, retrieved_contexts = rag.query(
        query=query, 
        source=source, # The wrapper handles the mapping!
        retrieve_top_k=eval_settings.retrieve_top_k, 
        rerank_top_k=eval_settings.rerank_top_k, # Set it to 0 to isolate retrieval performance without the reranker influence
        strategy="hybrid"
    )
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets
    )
    
    # CRITICAL FIX: Instantiate metric inside the test case to prevent state leakage
    recall_metric = ContextualRecallMetric(
        threshold=eval_settings.recall_threshold, 
        model=eval_judge, 
        include_reason=True
    )
    
    assert_test(test_case, [recall_metric], run_async=False)