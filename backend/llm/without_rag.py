"""
without_rag.py
--------------
LLM answer generation WITHOUT retrieval context.
Baseline comparison — LLM answers from training knowledge only.
"""

from __future__ import annotations

import logging
import time
import os

import ollama

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

client = ollama.Client(host=OLLAMA_HOST)

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support analyst.
When answering, you:
1. Give actionable, specific advice based on your knowledge
2. Acknowledge if the situation seems urgent
3. Keep responses concise and helpful (3-5 sentences)
Never mention that you are an AI."""

# ── Answer Generation ─────────────────────────────────────────────────────────
def answer_without_rag(query: str) -> dict:
    """
    Generate LLM answer using only the user query — no retrieved context.
    Used as baseline comparison against RAG answer.

    Args:
        query: User's support question

    Returns:
        dict with keys:
            - answer: LLM generated response
            - llm_latency_ms: time for LLM to generate answer
            - model: LLM model used
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    logger.info("Generating non-RAG answer for: '%s...'", query[:50])

    try:
        # Build prompt — no context, just the query
        user_message = f"""Customer query: {query}

Please provide a helpful response."""

        # Generate LLM response
        llm_start = time.time()
        response = client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        llm_latency = round((time.time() - llm_start) * 1000, 2)

        answer = response['message']['content']

        logger.info("Non-RAG answer generated in %sms", llm_latency)

        return {
            "answer": answer,
            "llm_latency_ms": llm_latency,
            "model": LLM_MODEL
        }

    except Exception as e:
        logger.error("Non-RAG answer failed: %s", e)
        raise RuntimeError(f"Non-RAG answer generation failed: {e}") from e