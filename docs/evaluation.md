# Evaluation Methodology

The evaluation suite uses [DeepEval](https://github.com/confident-ai/deepeval) to benchmark retrieval accuracy, factual faithfulness, and output precision against a ground-truth dataset ([golden_dataset_v2.json](file:///Users/r/Github/Insurance-RAG/evaluations/datasets/golden_dataset_v2.json)).

## 1. Metrics and Methodology

The test suite evaluates standard retrieval-augmented generation metrics alongside legal reasoning requirements:

| Metric | Target | Description | Test File |
| :--- | :--- | :--- | :--- |
| Answer Correctness | $\ge 0.70$ | Validates whether the final verdict matches ground-truth legal outcomes. | `test_standard_rag_metrics.py` |
| Reasoning Faithfulness | $\ge 0.90$ | Detects hallucinations by verifying that cited clauses and justifications exist directly in retrieved context. | `test_standard_rag_metrics.py` |
| Contextual Precision | $\ge 0.80$ | Evaluates cross-encoder effectiveness by verifying that ground-truth clauses are prioritized in the context window. | `test_reranker_impact.py` |
| Contextual Recall | $\ge 0.80$ | Evaluates hybrid retrieval by verifying that semantic search and BM25 retrieve all relevant clauses. | `test_hybrid_vs_semantic.py` |
| Answer Relevancy | $\ge 0.80$ | Measures conciseness and verifies that responses directly address inquiries without extraneous text. | `test_standard_rag_metrics.py` |

## 2. LLM Judge Configuration

Evaluations use an LLM judge (`GroqJudge`, backed by Llama 3.3 70B). Batch evaluation workloads are prone to rate limiting (HTTP 429).

To ensure deterministic benchmark runs, `evaluations/utils/custom_judge.py` wraps model invocations with exponential backoff retries using Tenacity:

```python
@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(10),
)
async def a_generate(self, prompt: str) -> str:
    res = await self.model.ainvoke(prompt)
    return res.content
```

## 3. Running Test Suites

Install dependencies with the `[dev]` optional package:

```bash
pip install -e ".[dev]"
```

Configure credentials in `.env` or `evaluations/eval_config.py`:
```env
GROQ_API_KEY=your_groq_key
DATASET_START_INDEX=0
DATASET_END_INDEX=5
```

Execute tests:

```bash
# Full test suite (Correctness, Faithfulness, Relevancy, Precision, Recall)
pytest evaluations/test_cases/test_standard_rag_metrics.py -v

# Reranker evaluation
pytest evaluations/test_cases/test_reranker_impact.py -v

# Hybrid vs. Dense retrieval comparison
pytest evaluations/test_cases/test_hybrid_vs_semantic.py -v
```
