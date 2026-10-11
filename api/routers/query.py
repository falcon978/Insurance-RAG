"""
api/routers/query.py
--------------------
Defines query endpoints for single-policy and cross-policy comparison RAG workflows.
"""

from fastapi import APIRouter, Depends, HTTPException

from engine import InsuranceRAGEngine
from api.dependencies import get_rag_engine
from schemas import (
    SingleQueryRequest,
    CompareQueryRequest,
    StandardResponse,
)

router = APIRouter(prefix="/api/v1/query", tags=["Query"])


@router.post("/single", response_model=StandardResponse)
async def query_single(
    req: SingleQueryRequest,
    rag_engine: InsuranceRAGEngine = Depends(get_rag_engine),
):
    """Processes a user inquiry against a single insurance policy collection."""
    try:
        answer = await rag_engine.a_query_single_policy(
            query=req.query,
            collection_name=req.collection_name,
            history=req.history,
            retrieve_top_k=req.retrieve_top_k,
            rerank_top_k=req.rerank_top_k,
        )
        return StandardResponse(
            status="success",
            message="Query processed",
            data={"markdown_report": answer},
        )
    except HTTPException:
        raise


@router.post("/compare", response_model=StandardResponse)
async def query_compare(
    req: CompareQueryRequest,
    rag_engine: InsuranceRAGEngine = Depends(get_rag_engine),
):
    """Executes a comparative policy evaluation between two policy collections."""
    try:
        answer = await rag_engine.a_compare_policies(
            query=req.query,
            collection_a=req.collection_a,
            collection_b=req.collection_b,
            history=req.history,
            retrieve_top_k=req.retrieve_top_k,
            rerank_top_k=req.rerank_top_k,
        )
        return StandardResponse(
            status="success",
            message="Comparison processed",
            data={"markdown_report": answer},
        )
    except HTTPException:
        raise
