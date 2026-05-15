"""
test_standard_rag_metrics.py
----------------------------
Executes the core RAG Triad (Recall, Precision, Faithfulness, Relevancy) + Custom Logic Metric
across the entire golden dataset.
"""

import pytest
import asyncio
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

from evaluations.utils.rag_wrapper import EvalRAGWrapper
from evaluations.datasets.data_loader import load_golden_dataset
from evaluations.metrics.insurance_metrics import (
    get_reasoning_faithfulness_metric,
    get_answer_correctness_metric,
)
from evaluations.eval_config import eval_settings
from evaluations.utils.custom_judge import get_eval_judge

rag = EvalRAGWrapper()
eval_judge = get_eval_judge()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id, query, source, expected_snippets, keywords, reasoning",
    load_golden_dataset(),
)
async def test_full_rag_triad(
    case_id, query, source, expected_snippets, keywords, reasoning
):
    await asyncio.sleep(eval_settings.rate_limit_delay_seconds)

    actual_output, retrieved_contexts = await rag.a_query(
        query=query,
        source=source,
        retrieve_top_k=eval_settings.retrieve_top_k,
        rerank_top_k=eval_settings.rerank_top_k,
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts,
        expected_retrieval_context=expected_snippets,
    )

    recall_metric = ContextualRecallMetric(
        threshold=eval_settings.recall_threshold, model=eval_judge, include_reason=True
    )
    precision_metric = ContextualPrecisionMetric(
        threshold=eval_settings.precision_threshold,
        model=eval_judge,
        include_reason=True,
    )
    faithfulness_metric = FaithfulnessMetric(
        threshold=eval_settings.faithfulness_threshold,
        model=eval_judge,
        include_reason=True,
    )
    relevancy_metric = AnswerRelevancyMetric(
        threshold=eval_settings.relevancy_threshold,
        model=eval_judge,
        include_reason=True,
    )

    correctness_metric = get_answer_correctness_metric()
    reasoning_metric = get_reasoning_faithfulness_metric()

    assert_test(
        test_case,
        [
            recall_metric,
            precision_metric,
            faithfulness_metric,
            relevancy_metric,
            correctness_metric,
            reasoning_metric,
        ],
        run_async=False,
    )
