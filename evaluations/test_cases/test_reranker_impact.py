"""
test_reranker_impact.py
-----------------------
A/B testing for Contextual Precision to measure how much value the cross-encoder
adds compared to raw vector search ranking.
"""

import pytest
import time
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

from evaluations.utils.rag_wrapper import EvalRAGWrapper
from evaluations.datasets.data_loader import load_golden_dataset
from evaluations.eval_config import eval_settings
from evaluations.utils.custom_judge import get_eval_judge

rag = EvalRAGWrapper()
eval_judge = get_eval_judge()


@pytest.mark.parametrize(
    "case_id, query, source, expected_snippets, keywords, reasoning",
    load_golden_dataset(),
)
def test_baseline_no_reranker(
    case_id, query, source, expected_snippets, keywords, reasoning
):

    time.sleep(eval_settings.rate_limit_delay_seconds)

    # Top 3 directly from Chroma/Pinecone (No Reranker)
    actual_output, retrieved_contexts = rag.query(
        query=query,
        source=source,  # The wrapper handles the mapping!
        retrieve_top_k=eval_settings.retrieve_top_k,
        rerank_top_k=0,
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets,
    )

    # CRITICAL FIX: Instantiate metric inside the test case to prevent state leakage
    precision_metric = ContextualPrecisionMetric(
        threshold=eval_settings.precision_threshold,
        model=eval_judge,
        include_reason=True,
    )

    assert_test(test_case, [precision_metric], run_async=False)


@pytest.mark.parametrize(
    "case_id, query, source, expected_snippets, keywords, reasoning",
    load_golden_dataset(),
)
def test_advanced_with_reranker(
    case_id, query, source, expected_snippets, keywords, reasoning
):

    time.sleep(eval_settings.rate_limit_delay_seconds)

    # Top 15 narrowed down to Top 3 via Reranker
    actual_output, retrieved_contexts = rag.query(
        query=query,
        source=source,  # The wrapper handles the mapping!
        retrieve_top_k=eval_settings.retrieve_top_k,
        rerank_top_k=eval_settings.rerank_top_k,
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets,
    )

    # CRITICAL FIX: Instantiate metric inside the test case to prevent state leakage
    precision_metric = ContextualPrecisionMetric(
        threshold=eval_settings.precision_threshold,
        model=eval_judge,
        include_reason=True,
    )

    assert_test(test_case, [precision_metric], run_async=False)
