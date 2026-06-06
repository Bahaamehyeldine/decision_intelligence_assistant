"""
output.py
---------
Pydantic schemas for API response validation.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class RAGResponse(BaseModel):
    answer: str
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    low_similarity: bool
    context_count: int


class NonRAGResponse(BaseModel):
    answer: str
    llm_latency_ms: float


class MLPrediction(BaseModel):
    priority: int
    priority_label: str
    confidence: float
    urgent_probability: float
    latency_ms: float


class LLMZeroShot(BaseModel):
    priority: int
    priority_label: str
    latency_ms: float


class QueryResponse(BaseModel):
    query: str
    rag: Optional[RAGResponse] = None
    non_rag: Optional[NonRAGResponse] = None
    ml_prediction: Optional[MLPrediction] = None
    llm_zeroshot: Optional[LLMZeroShot] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    qdrant: dict
    ollama: str
    ml_model: str