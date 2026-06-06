"""
routes.py
---------
FastAPI route definitions for the Decision Intelligence Assistant.
Four endpoints: /query, /health, /logs, /stats
"""

from __future__ import annotations

import logging
from typing import Optional

import ollama
from fastapi import APIRouter, HTTPException

from llm.with_rag import answer_with_rag
from llm.without_rag import answer_without_rag
from ml.predict import predict_priority
from rag.vectorstore import health_check as qdrant_health
from logs.logger import log_query, get_recent_logs, get_log_stats
from schemas.input import QueryRequest
from schemas.output import QueryResponse, HealthResponse

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter()

# ── LLM Zero-shot Priority Prediction ────────────────────────────────────────
OLLAMA_HOST = __import__('os').getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)
LLM_MODEL = __import__('os').getenv("LLM_MODEL", "llama3.2")


def llm_zeroshot_priority(query: str) -> dict:
    """
    Ask LLM directly if ticket is urgent or normal.
    Zero-shot — no training, no context, just the query.
    """
    import time
    start = time.time()

    prompt = f"""Classify this customer support ticket as either 'urgent' or 'normal'.

Ticket: "{query}"

Reply with ONLY one word: urgent or normal"""

    response = ollama_client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response['message']['content'].strip().lower()
    priority_label = "urgent" if "urgent" in answer else "normal"
    priority = 1 if priority_label == "urgent" else 0
    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "priority": priority,
        "priority_label": priority_label,
        "latency_ms": latency_ms
    }


# ── POST /query ───────────────────────────────────────────────────────────────
@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main endpoint — processes query through all four systems:
    1. RAG answer (LLM + retrieved context)
    2. Non-RAG answer (LLM only)
    3. ML priority prediction
    4. LLM zero-shot priority prediction
    """
    logger.info("Processing query: '%s...'", request.query[:50])
    error = None

    rag_result = None
    non_rag_result = None
    ml_result = None
    llm_zeroshot_result = None

    # Run all four systems — don't fail if one crashes
    try:
        rag_result = answer_with_rag(request.query, top_k=request.top_k)
    except Exception as e:
        logger.error("RAG failed: %s", e)
        error = f"RAG error: {str(e)}"

    try:
        non_rag_result = answer_without_rag(request.query)
    except Exception as e:
        logger.error("Non-RAG failed: %s", e)
        error = f"Non-RAG error: {str(e)}"

    try:
        ml_result = predict_priority(request.query)
    except Exception as e:
        logger.error("ML prediction failed: %s", e)
        error = f"ML error: {str(e)}"

    try:
        llm_zeroshot_result = llm_zeroshot_priority(request.query)
    except Exception as e:
        logger.error("LLM zero-shot failed: %s", e)
        error = f"LLM zero-shot error: {str(e)}"

    # Log everything
    log_query(
        query=request.query,
        rag_result=rag_result,
        non_rag_result=non_rag_result,
        ml_result=ml_result,
        llm_zeroshot_result=llm_zeroshot_result,
        error=error
    )

    # Build response
    return QueryResponse(
        query=request.query,
        rag={
            "answer": rag_result.get("answer"),
            "retrieval_latency_ms": rag_result.get("retrieval_latency_ms"),
            "llm_latency_ms": rag_result.get("llm_latency_ms"),
            "total_latency_ms": rag_result.get("total_latency_ms"),
            "low_similarity": rag_result.get("low_similarity"),
            "context_count": len(rag_result.get("context_used", []))
        } if rag_result else None,
        non_rag=non_rag_result,
        ml_prediction=ml_result,
        llm_zeroshot=llm_zeroshot_result,
        error=error
    )


# ── GET /health ───────────────────────────────────────────────────────────────
@router.get("/health")
async def health_check():
    """Check status of all services."""
    qdrant_status = qdrant_health()

    # Check Ollama
    try:
        ollama_client.list()
        ollama_status = "healthy"
    except Exception as e:
        ollama_status = f"unhealthy: {str(e)}"

    # Check ML model
    try:
        from ml.predict import model
        ml_status = f"healthy ({type(model).__name__})"
    except Exception as e:
        ml_status = f"unhealthy: {str(e)}"

    overall = "healthy" if (
        qdrant_status.get("status") == "healthy" and
        ollama_status == "healthy" and
        "healthy" in ml_status
    ) else "degraded"

    return {
        "status": overall,
        "qdrant": qdrant_status,
        "ollama": ollama_status,
        "ml_model": ml_status
    }


# ── GET /logs ─────────────────────────────────────────────────────────────────
@router.get("/logs")
async def get_logs(n: int = 10):
    """Get recent query logs."""
    logs = get_recent_logs(n=n)
    return {"logs": logs, "count": len(logs)}


# ── GET /stats ────────────────────────────────────────────────────────────────
@router.get("/stats")
async def get_stats():
    """Get aggregate statistics from all logged queries."""
    return get_log_stats()