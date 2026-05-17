"""
test_hybrid_vs_semantic.py
--------------------------
A/B testing for Contextual Recall.
Measures the retrieval performance of Pure Vector Search vs. Hybrid Search (Vector + BM25).
"""

import pytest
import pytest_asyncio  # <-- ADDED for async fixtures
import asyncio
import redis.asyncio as redis
from pinecone import Pinecone
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

from evaluations.utils.rag_wrapper import EvalRAGWrapper
from evaluations.datasets.data_loader import load_golden_dataset
from evaluations.eval_config import eval_settings
from evaluations.utils.custom_judge import get_eval_judge

rag = EvalRAGWrapper()
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


# ---------------------------------------------------------
# TEST A: Pure Semantic Baseline
# ---------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id, query, source, expected_snippets, keywords, reasoning",
    load_golden_dataset(),
)
async def test_pure_semantic_search(
    case_id, query, source, expected_snippets, keywords, reasoning
):
    await asyncio.sleep(eval_settings.rate_limit_delay_seconds)

    actual_output, retrieved_contexts = await rag.a_query(
        query=query,
        source=source,
        retrieve_top_k=eval_settings.retrieve_top_k,
        rerank_top_k=0,
        strategy="semantic",
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

    assert_test(test_case, [recall_metric], run_async=False)


# ---------------------------------------------------------
# TEST B: Hybrid Search (Semantic + BM25)
# ---------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id, query, source, expected_snippets, keywords, reasoning",
    load_golden_dataset(),
)
async def test_hybrid_search(
    case_id, query, source, expected_snippets, keywords, reasoning
):
    await asyncio.sleep(eval_settings.rate_limit_delay_seconds)

    actual_output, retrieved_contexts = await rag.a_query(
        query=query,
        source=source,
        retrieve_top_k=eval_settings.retrieve_top_k,
        rerank_top_k=0,
        strategy="hybrid",
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

    assert_test(test_case, [recall_metric], run_async=False)
