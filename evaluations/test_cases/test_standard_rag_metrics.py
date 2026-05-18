"""
test_standard_rag_metrics.py
----------------------------
Executes the core RAG Triad (Recall, Precision, Faithfulness, Relevancy) + Custom Logic Metric
across the entire golden dataset.
"""

import pytest
import pytest_asyncio  # <-- ADDED for async fixtures
import asyncio
import redis.asyncio as redis
from pinecone import Pinecone

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

eval_judge = get_eval_judge()


# --- CONNECTION POOL FIXTURE ---
@pytest_asyncio.fixture(scope="module")
async def rag_wrapper():
    """Spins up a shared database connection pool for all tests in this file."""

    # 1. Initialize Redis TCP Pool (allow slightly higher max_connections for async testing)
    redis_client = redis.from_url(
        eval_settings.redis_url,
        decode_responses=True,
        health_check_interval=30,
        retry_on_timeout=True,
        max_connections=50,
    )

    # 2. Initialize Pinecone HTTP Session
    pc = Pinecone(api_key=eval_settings.pinecone_api_key)
    pinecone_index = pc.Index(eval_settings.pinecone_index_name)

    # 3. Inject into the wrapper
    wrapper = EvalRAGWrapper(redis_client=redis_client, pinecone_index=pinecone_index)

    yield wrapper

    # 4. Graceful Teardown after all tests finish
    await redis_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id, query, source, expected_snippets, keywords, reasoning",
    load_golden_dataset(),
)
async def test_full_rag_triad(
    case_id,
    query,
    source,
    expected_snippets,
    keywords,
    reasoning,
    rag_wrapper,  # <-- INJECT FIXTURE HERE
):
    await asyncio.sleep(eval_settings.rate_limit_delay_seconds)

    # Call a_query on the injected wrapper instance
    actual_output, retrieved_contexts = await rag_wrapper.a_query(
        query=query,
        source=source,
        retrieve_top_k=eval_settings.retrieve_top_k,
        rerank_top_k=eval_settings.rerank_top_k,
    )

    expected_output_str = (
        f"Keywords to include: {', '.join(keywords)}. Logic: {reasoning}"
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output_str,
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
            recall_metric,  # Evaluates embedding model (Recall@K)
            precision_metric,  # Evaluates reranker/ranking (MRR)
            faithfulness_metric,  # Evaluates LLM hallucination
            relevancy_metric,  # Evaluates generic answer quality
            correctness_metric,  # Did the LLM reach the correct legal outcome?
            reasoning_metric,  # Did the LLM use the right policy logic?
        ],
        run_async=False,  # Forces metrics to evaluate one-by-one
    )
