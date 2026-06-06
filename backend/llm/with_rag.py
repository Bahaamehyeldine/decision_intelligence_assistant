"""
with_rag.py
-----------
LLM answer generation using RAG context.
Retrieves similar past tickets and injects them into the prompt
before asking the LLM to generate a response.
"""

from __future__ import annotations

import logging
import time
import os
from typing import Optional

import ollama

from rag.retrieval import retrieve, build_rag_context

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
TOP_K = 5

client = ollama.Client(host=OLLAMA_HOST)


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support analyst with access to 
a database of past support tickets. When answering, you:
1. Reference specific patterns from the similar tickets provided
2. Give actionable, specific advice based on real cases
3. Acknowledge if the situation seems urgent based on past cases
4. Keep responses concise and helpful (3-5 sentences)
Never mention that you are an AI or that you are using past tickets."""


# ── RAG Answer Generation ─────────────────────────────────────────────────────
def answer_with_rag(
    query: str,
    top_k: int = TOP_K,
) -> dict:
    """
    Generate LLM answer using retrieved similar tickets as context.

    Args:
        query: User's support question
        top_k: Number of similar tickets to retrieve

    Returns:
        dict with keys:
            - answer: LLM generated response
            - context_used: retrieved tickets used as context
            - retrieval_latency_ms: time to retrieve tickets
            - llm_latency_ms: time for LLM to generate answer
            - total_latency_ms: total response time
            - low_similarity: whether retrieved tickets were relevant
            - model: LLM model used
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    logger.info("Generating RAG answer for: '%s...'", query[:50])
    total_start = time.time()

    try:
        # Step 1: Retrieve similar tickets
        retrieval_result = retrieve(query, top_k=top_k)
        retrieval_latency = retrieval_result["latency_ms"]
        context = build_rag_context(retrieval_result["results"])

        # Step 2: Build prompt with context
        user_message = f"""Similar past support tickets:
{context}

Current customer query: {query}

Based on the similar cases above, provide a helpful response."""

        # Step 3: Generate LLM response
        llm_start = time.time()
        response = client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        llm_latency = round((time.time() - llm_start) * 1000, 2)
        total_latency = round((time.time() - total_start) * 1000, 2)

        answer = response['message']['content']

        logger.info(
            "RAG answer generated. Retrieval: %sms, LLM: %sms, Total: %sms",
            retrieval_latency, llm_latency, total_latency
        )

        return {
            "answer": answer,
            "context_used": retrieval_result["results"],
            "retrieval_latency_ms": retrieval_latency,
            "llm_latency_ms": llm_latency,
            "total_latency_ms": total_latency,
            "low_similarity": retrieval_result["low_similarity"],
            "model": LLM_MODEL
        }

    except Exception as e:
        logger.error("RAG answer failed: %s", e)
        raise RuntimeError(f"RAG answer generation failed: {e}") from e