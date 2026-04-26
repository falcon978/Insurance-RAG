# evaluations/test_cases/test_hybrid_vs_semantic.py
import pytest
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

from evaluations.utils.rag_wrapper import EvalRAGWrapper
from evaluations.datasets.data_loader import load_golden_dataset

rag = EvalRAGWrapper()
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")

# Recall checks if the required documents were fetched at all
recall_metric = ContextualRecallMetric(threshold=0.8, model=JUDGE_MODEL, include_reason=True)

@pytest.mark.parametrize("case_id, query, source, expected_snippets, keywords, reasoning", load_golden_dataset())
def test_pure_semantic_search(case_id, query, source, expected_snippets, keywords, reasoning):
    collection = "hdfc_care_docs" if source == "both" else f"{source}_docs"
    
    # disable_bm25=True forces pure vector search
    actual_output, retrieved_contexts = rag.query(
        query=query, collection_name=collection, retrieve_top_k=10, rerank_top_k=0, disable_bm25=True
    )
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets
    )
    
    assert_test(test_case, [recall_metric])