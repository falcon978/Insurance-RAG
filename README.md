# Insurance-RAG 

An advanced, production-grade Retrieval-Augmented Generation (RAG) system built to accurately analyze, query, and compare complex insurance policies. 

Unlike standard RAG pipelines that often hallucinate or miss critical legal nuances, this system is specifically engineered for high-stakes document analysis. It utilizes parallel hybrid search, cross-encoder reranking, and deterministic Chain-of-Thought LLM generation to ensure every answer is grounded, factual, and strictly adheres to policy inclusions and exclusions.

**Live Demo:** [Streamlit](https://insurance-rag-978.streamlit.app/)

---

## 🌟 Overview (For Product & Business Teams)

Reading insurance policies is tedious and error-prone. This application solves that by allowing users to upload policy documents and ask plain-English questions (e.g., *"Is a pre-existing ACL tear covered?"* or *"Compare the maternity benefits between these two policies"*). 

**Key Benefits:**
* **Zero Hallucinations:** The AI is mathematically constrained to only answer based on the provided text. If a policy is ambiguous or silent on an issue, the system explicitly flags it rather than guessing.
* **Legal Logic Emulation:** The system evaluates documents in a strict hierarchy: *Specific Exceptions -> General Exclusions -> Explicit Coverage -> Ambiguity*.
* **Deep Comparisons:** Capable of analyzing two different policies side-by-side to highlight coverage gaps and premium-to-benefit ratios.

---

## ⚙️ Technical Architecture (For Engineering Teams)

This project was developed with a microservices-ready architecture, prioritizing retrieval accuracy, strict prompt engineering, and low-latency inference on CPU-bound environments. 

### Core Pipeline Components
1. **Hierarchical Ingestion:** Extracts text from PDFs while preserving document structures (Sections -> Clauses -> Sub-clauses) using Markdown markers.
2. **Parallel Hybrid Search:** Translates user queries into optimized legal/medical terms. It simultaneously fires a Vector Search (Pinecone) for semantic meaning and a BM25 Search (Redis) for exact legal phrasing matching.
3. **Reciprocal Rank Fusion (RRF):** Merges the results of the semantic and lexical searches mathematically to surface the most robust candidate chunks.
4. **Lightweight Cross-Encoder Reranking:** Passes the fused results through `ms-marco-MiniLM-L-12-v2`. This model was specifically chosen to drastically improve contextual precision while keeping latency exceptionally low on standard Hugging Face CPU tiers.
5. **Deterministic Generation (Gemini):** Utilizes a custom Sentinel Output Parser. The LLM is forced to output its internal reasoning in strict JSON format before generating the user-facing Markdown report. This creates a highly auditable Chain-of-Thought.
6. **Observability & Tracing:** Integrated with **LangSmith** to capture full execution traces, allowing detailed debugging of the prompt engineering steps, multi-stage retrieval paths, and reranking latencies.

### Tech Stack
* **Backend:** FastAPI, Python
* **Frontend:** Streamlit (`app.py`, `eval_dashboard.py`)
* **AI / Models:** Google Gemini (Generation), ms-marco-MiniLM (Cross-Encoder Reranker), BAAI/bge-large-en (Embeddings)
* **Databases:** Pinecone (Vector Store), Redis (BM25 Lexical Store)
* **Testing:** DeepEval (Correctness & Reasoning Faithfulness metrics)

---

## 📂 Repository Structure

* `app.py`: Streamlit UI entry point
* `main.py`: FastAPI backend orchestrator & endpoints
* `config.py`: Central environment and model configurations
* `rag_ingestion/`: PDF extraction, cleaning, hierarchical chunking, and indexing
* `rag/`: Retrieval engine, prompt templates, rerankers, and LLM generators
* `evaluations/`: DeepEval testing suite and custom insurance metrics
* `notebooks/`: Jupyter notebooks for pipeline walk-throughs and A/B testing
* `Dockerfile`: Containerization for Hugging Face Spaces

---

## 🚀 Getting Started (Local Development)

### Prerequisites
* Python 3.10+
* Redis server running locally or via Docker
* Pinecone API Key
* Google Gemini API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/insurance-rag.git](https://github.com/yourusername/insurance-rag.git)
   cd insurance-rag

# Setup

## 1. Set up a Virtual Environment

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Environment Variables

Create a `.env` file in the project root and add the following:

```env
PINECONE_API_KEY=your_key
GEMINI_API_KEY=your_key
REDIS_URL=redis://localhost:6379
```

The system relies on a highly configurable environment architecture, managed via Pydantic BaseSettings. Using the .env file, you can easily experiment with different providers, models, and runtime settings without modifying the codebase.

## 4. Run the Backend (FastAPI)

```bash
uvicorn main:app --reload
```

## 5. Run the Frontend (Streamlit)

```bash
streamlit run app.py
```

---

# 🧪 Evaluation & Testing

This project includes an automated evaluation pipeline built with **DeepEval** to comprehensively test both the retrieval and generation stages of the RAG lifecycle. 

The test suite evaluates the system across multiple parameters:
* **Answer Correctness:** Does the final answer accurately resolve the user's query based on the golden dataset?
* **Reasoning Faithfulness (Hallucination Detection):** Is the generated reasoning and final response strictly derived from the retrieved context without fabricating information?
* **Contextual Precision:** Evaluates the reranking stage (`test_reranker_impact.py`) to ensure the most relevant policy chunks are pushed to the top of the context window.
* **Contextual Recall:** Evaluates the retrieval strategy (`test_hybrid_vs_semantic.py`) to ensure the hybrid search successfully retrieves all necessary clauses and exclusions required to answer the query.
* **Answer Relevancy:** Ensures the response is direct, concise, and free of redundant or off-topic information.

To run the evaluation suite against the golden dataset:

```bash
pytest evaluations/test_cases/
```
