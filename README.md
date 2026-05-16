---
title: Insurance RAG
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Insurance RAG: Legal-Grade Policy Analysis Engine

An advanced Retrieval-Augmented Generation (RAG) system designed to parse, index, and analyze complex insurance policy documents with high legal accuracy. This system utilizes a **Unified Single-Pass Architecture** to provide conversational clarity while maintaining strict adherence to policy clauses, waiting periods, and exclusions.

## 🏗️ Architecture Overview

The project is built on a decoupled, microservices-ready architecture:

* **Frontend**: A Streamlit-based dashboard providing conversational chat with sliding window memory and an admin panel for collection management.
* **Backend**: A FastAPI orchestration layer managing ingestion pipelines and RAG query execution.
* **Ingestion Pipeline**: A multi-stage engine using `PyMuPDF` for layout-aware extraction, hierarchical Markdown chunking, and dual-vector/lexical indexing.
* **RAG Engine**: A hybrid search system combining semantic embeddings with BM25 lexical search, followed by a Cross-Encoder reranking stage.

## 🚀 Key Features

### 1. Hybrid Search & Reranking

* **Dual Retrieval**: Combines semantic meaning via `BAAI/bge-large-en-v1.5` with keyword precision using `BM25Retriever`.
* **Reciprocal Rank Fusion (RRF)**: Merges retrieval streams using weighted scoring to deduplicate and re-score documents.
* **Cross-Encoder Reranking**: Utilizes `BGE-Reranker-v2-m3` to score the top-K candidates against the query for maximum contextual relevance.

### 2. Unified Single-Pass Generation

* **Sentinel Extraction**: Employs a custom `SentinelOutputParser` using strict `<<<BEGIN>>>` tags to separate internal "Adjudication JSON" (Chain-of-Thought) from the user-facing "Advisory Report".
* **Deterministic Logic**: Powered by Google Gemini with a temperature of 0.0 to ensure consistent, evidence-based responses.

### 3. Automated Evaluation Suite

* **DeepEval Integration**: Uses G-Eval metrics to assess "Answer Correctness" and "Reasoning Faithfulness" independently.
* **Double-Citation Mandate**: Ensures every claim in the generated report is backed by specific snippets from the policy documents.

## 🛠️ Tech Stack

* **Frameworks**: FastAPI, Streamlit, LangChain.
* **LLMs**: Google Gemini (3-Flash-Preview & 3.1-Flash-Lite).
* **Vector DBs**: ChromaDB (Local) or Pinecone (Cloud).
* **Embeddings**: HuggingFace Sentence-Transformers.

## 📥 Getting Started

### Prerequisites

* Python 3.10+
* Google Gemini API Key

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

```

### Configuration

Create a `.env` file or export variables based on `config.py`:

```bash
GEMINI_API_KEY=your_key_here
VECTOR_DB_TYPE=chroma  # or 'pinecone'
HF_DEVICE=cpu          # or 'cuda' for GPU acceleration

```

### Running the System

1. **Start the Backend API**:
```bash
uvicorn main:app --reload --port 8000

```


2. **Start the Frontend Client**:
```bash
streamlit run app.py

```



## 📂 Project Structure

* `main.py`: FastAPI endpoints for ingestion and querying.
* `app.py`: Streamlit UI for chat and admin management.
* `rag/`: Core logic for retrieval, reranking, and generation.
* `rag_ingestion/`: PDF extraction and hierarchical chunking pipelines.
* `evaluations/`: DeepEval metrics and test cases for pipeline validation.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/improvement`).
3. Commit your changes (`git commit -am 'Add improvement'`).
4. Push to the branch (`git push origin feature/improvement`).
5. Submit a Pull Request.

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.