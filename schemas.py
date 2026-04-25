"""
schemas.py
----------
Pydantic schemas for FastAPI request and response validation.
"""
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict

class SingleQueryRequest(BaseModel):
    query: str
    collection_name: str
    # History format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    history: Optional[List[Dict[str, str]]] = []
    retrieve_top_k: int = 15
    rerank_top_k: int = 3

class CompareQueryRequest(BaseModel):
    query: str
    collection_a: str
    collection_b: str
    history: Optional[List[Dict[str, str]]] = []
    retrieve_top_k: int = 15
    rerank_top_k: int = 3

class UrlIngestRequest(BaseModel):
    url: HttpUrl
    collection_name: str

class CollectionListResponse(BaseModel):
    collections: List[str]

class StandardResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None