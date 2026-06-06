"""
logger.py
---------
Structured query logging for the Decision Intelligence Assistant.
Logs every query with all four system outputs, latencies, and costs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── Log File Setup ────────────────────────────────────────────────────────────
LOG_DIR = Path(os.getenv("LOG_DIR", "/app/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "queries.jsonl"

# ── LLM Cost Estimation ───────────────────────────────────────────────────────
# Ollama runs locally — cost is effectively $0
# But we estimate compute cost for comparison purposes
COST_PER_MS = 0.000001  # $0.000001 per ms of LLM compute


def estimate_llm_cost(latency_ms: float) -> float:
    """Estimate LLM compute cost based on latency."""
    return round(latency_ms * COST_PER_MS, 8)


# ── Query Logger ──────────────────────────────────────────────────────────────
def log_query(
    query: str,
    rag_result: Optional[dict] = None,
    non_rag_result: Optional[dict] = None,
    ml_result: Optional[dict] = None,
    llm_zeroshot_result: Optional[dict] = None,
    error: Optional[str] = None
) -> dict:
    """
    Log a complete query interaction with all system outputs.

    Args:
        query: User's original query
        rag_result: Output from answer_with_rag()
        non_rag_result: Output from answer_without_rag()
        ml_result: Output from predict_priority()
        llm_zeroshot_result: Output from LLM zero-shot prediction
        error: Error message if any system failed

    Returns:
        Complete log entry as dict
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build log entry
    entry = {
        "timestamp": timestamp,
        "query": query,
        "query_length": len(query.split()),

        # RAG system
        "rag": {
            "answer": rag_result.get("answer") if rag_result else None,
            "retrieval_latency_ms": rag_result.get("retrieval_latency_ms") if rag_result else None,
            "llm_latency_ms": rag_result.get("llm_latency_ms") if rag_result else None,
            "total_latency_ms": rag_result.get("total_latency_ms") if rag_result else None,
            "low_similarity": rag_result.get("low_similarity") if rag_result else None,
            "context_count": len(rag_result.get("context_used", [])) if rag_result else 0,
            "estimated_cost": estimate_llm_cost(
                rag_result.get("llm_latency_ms", 0)) if rag_result else 0
        },

        # Non-RAG system
        "non_rag": {
            "answer": non_rag_result.get("answer") if non_rag_result else None,
            "llm_latency_ms": non_rag_result.get("llm_latency_ms") if non_rag_result else None,
            "estimated_cost": estimate_llm_cost(
                non_rag_result.get("llm_latency_ms", 0)) if non_rag_result else 0
        },

        # ML classifier
        "ml_classifier": {
            "priority": ml_result.get("priority") if ml_result else None,
            "priority_label": ml_result.get("priority_label") if ml_result else None,
            "confidence": ml_result.get("confidence") if ml_result else None,
            "urgent_probability": ml_result.get("urgent_probability") if ml_result else None,
            "latency_ms": ml_result.get("latency_ms") if ml_result else None,
            "estimated_cost": 0.0  # ML is free — runs locally
        },

        # LLM zero-shot
        "llm_zeroshot": {
            "priority": llm_zeroshot_result.get("priority") if llm_zeroshot_result else None,
            "priority_label": llm_zeroshot_result.get("priority_label") if llm_zeroshot_result else None,
            "latency_ms": llm_zeroshot_result.get("latency_ms") if llm_zeroshot_result else None,
            "estimated_cost": estimate_llm_cost(
                llm_zeroshot_result.get("latency_ms", 0)) if llm_zeroshot_result else 0
        },

        # Error tracking
        "error": error,

        # Summary metrics
        "summary": {
            "ml_vs_llm_agreement": (
                ml_result.get("priority") == llm_zeroshot_result.get("priority")
                if ml_result and llm_zeroshot_result else None
            ),
            "total_llm_cost": round(
                estimate_llm_cost(rag_result.get("llm_latency_ms", 0) if rag_result else 0) +
                estimate_llm_cost(non_rag_result.get("llm_latency_ms", 0) if non_rag_result else 0) +
                estimate_llm_cost(llm_zeroshot_result.get("latency_ms", 0) if llm_zeroshot_result else 0),
                8
            )
        }
    }

    # Write to JSONL file
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        logger.info("Query logged: '%s...'", query[:50])
    except Exception as e:
        logger.error("Failed to write log entry: %s", e)

    return entry


# ── Log Reader ────────────────────────────────────────────────────────────────
def get_recent_logs(n: int = 10) -> list[dict]:
    """
    Read the most recent n log entries.

    Args:
        n: Number of recent entries to return

    Returns:
        List of log entry dicts, most recent last
    """
    try:
        if not LOG_FILE.exists():
            return []

        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()

        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

        return entries

    except Exception as e:
        logger.error("Failed to read logs: %s", e)
        return []


# ── Log Statistics ────────────────────────────────────────────────────────────
def get_log_stats() -> dict:
    """
    Compute statistics from all logged queries.

    Returns:
        dict with aggregate metrics
    """
    try:
        entries = get_recent_logs(n=10000)

        if not entries:
            return {"total_queries": 0}

        total = len(entries)
        rag_latencies = [
            e["rag"]["total_latency_ms"]
            for e in entries
            if e["rag"]["total_latency_ms"]
        ]
        ml_latencies = [
            e["ml_classifier"]["latency_ms"]
            for e in entries
            if e["ml_classifier"]["latency_ms"]
        ]
        agreements = [
            e["summary"]["ml_vs_llm_agreement"]
            for e in entries
            if e["summary"]["ml_vs_llm_agreement"] is not None
        ]

        return {
            "total_queries": total,
            "avg_rag_latency_ms": round(
                sum(rag_latencies) / len(rag_latencies), 2
            ) if rag_latencies else None,
            "avg_ml_latency_ms": round(
                sum(ml_latencies) / len(ml_latencies), 2
            ) if ml_latencies else None,
            "ml_llm_agreement_rate": round(
                sum(agreements) / len(agreements), 4
            ) if agreements else None,
            "total_estimated_cost": round(
                sum(e["summary"]["total_llm_cost"] for e in entries), 6
            )
        }

    except Exception as e:
        logger.error("Failed to compute log stats: %s", e)
        return {"error": str(e)}