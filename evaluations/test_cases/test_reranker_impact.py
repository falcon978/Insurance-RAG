# evaluations/test_cases/test_reranker_impact.py
import pytest
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

from evaluations.utils.rag_wrapper import EvalRAGWrapper
from evaluations.datasets.data_loader import load_golden_dataset

rag = EvalRAGWrapper()
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")

# We only care about Contextual Precision (ranking quality) for this test
precision_metric = ContextualPrecisionMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True)

@pytest.mark.parametrize("case_id, query, source, expected_snippets, keywords, reasoning", load_golden_dataset())
def test_baseline_no_reranker(case_id, query, source, expected_snippets, keywords, reasoning):
    collection = "hdfc_care_docs" if source == "both" else f"{source}_docs"
    
    # Top 5 directly from Chroma/Pinecone
    actual_output, retrieved_contexts = rag.query(
        query=query, collection_name=collection, retrieve_top_k=5, rerank_top_k=0
    )
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets
    )
    
    assert_test(test_case, [precision_metric])

@pytest.mark.parametrize("case_id, query, source, expected_snippets, keywords, reasoning", load_golden_dataset())
def test_advanced_with_reranker(case_id, query, source, expected_snippets, keywords, reasoning):
    collection = "hdfc_care_docs" if source == "both" else f"{source}_docs"
    
    # Top 15 narrowed down to Top 3 via Reranker
    actual_output, retrieved_contexts = rag.query(
        query=query, collection_name=collection, retrieve_top_k=15, rerank_top_k=3
    )
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets
    )
    
    assert_test(test_case, [precision_metric])