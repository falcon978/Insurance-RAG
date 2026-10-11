# API Reference

Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs`.

## 1. Query Endpoints

### `POST /api/v1/query/single`
Processes a single-policy inquiry with optional conversational history.

**Request Body:**
```json
{
  "query": "Is robotic knee replacement covered, and what is the waiting period?",
  "collection_name": "insurance_optima_secure",
  "history": [
    {"role": "user", "content": "Hello, I want to know about surgical benefits."},
    {"role": "assistant", "content": "I can help with surgical benefits under your policy."}
  ],
  "retrieve_top_k": 15,
  "rerank_top_k": 5
}
```

**Response (`StandardResponse`):**
```json
{
  "status": "success",
  "message": "Query processed",
  "data": {
    "markdown_report": "### Executive Summary\nRobotic knee replacement is covered under Modern Treatments..."
  }
}
```

### `POST /api/v1/query/compare`
Executes a comparative legal evaluation between two policy collections.

**Request Body:**
```json
{
  "query": "Compare maternity benefits, limits, and waiting periods.",
  "collection_a": "insurance_optima_secure",
  "collection_b": "insurance_care_supreme",
  "history": [],
  "retrieve_top_k": 15,
  "rerank_top_k": 5
}
```

## 2. Administration and Ingestion Endpoints

### `POST /api/v1/admin/ingest/file`
Uploads and indexes a policy PDF via `multipart/form-data`.

**Form Parameters:**
* `file`: PDF file binary.
* `collection_name`: Target collection identifier (for example, `insurance_optima_secure`).
* `chunk_size`: Integer (default: `2000`).
* `chunk_overlap`: Integer (default: `300`).

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/admin/ingest/file" \
     -F "file=@/path/to/policy.pdf" \
     -F "collection_name=insurance_optima_secure"
```

### `POST /api/v1/admin/ingest/url`
Downloads and indexes a policy PDF from an external URL.

**Request Body:**
```json
{
  "url": "https://example.com/policy_document.pdf",
  "collection_name": "insurance_care_supreme",
  "chunk_size": 2000,
  "chunk_overlap": 300
}
```

### `GET /api/v1/admin/collections`
Returns a list of all indexed collections across the active vector database.

**Response:**
```json
{
  "collections": [
    "insurance_optima_secure",
    "insurance_care_supreme"
  ]
}
```

### `DELETE /api/v1/admin/collections/{collection_name}`
Drops the specified collection from both vector and lexical persistent stores.

