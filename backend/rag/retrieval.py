"""
retrieval.py
------------
Semantic similarity search for the RAG system.
Encodes user queries and retrieves top-k similar support tickets from Qdrant.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from rag.embeddings import encode_text
from rag.vectorstore import client, COLLECTION_NAME

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3


# ── Search Result Model ───────────────────────────────────────────────────────
class SearchResult:
    """Represents a single retrieved support ticket."""

    def __init__(self, text: str, cleaned_text: str,
                 priority: int, score: float, tweet_id: int):
        self.text = text
        self.cleaned_text = cleaned_text
        self.priority = priority
        self.score = score
        self.tweet_id = tweet_id

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "cleaned_text": self.cleaned_text,
            "priority": self.priority,
            "similarity_score": round(self.score, 4),
            "tweet_id": self.tweet_id
        }

    def __repr__(self) -> str:
        return (f"SearchResult(score={self.score:.3f}, "
                f"priority={self.priority}, "
                f"text='{self.text[:50]}...')")


# ── Core Retrieval ────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    priority_filter: Optional[int] = None,
    min_score: float = SIMILARITY_THRESHOLD
) -> dict:
    """
    Retrieve top-k similar support tickets for a user query.

    Args:
        query: User's support question or complaint
        top_k: Number of similar tickets to retrieve
        priority_filter: If set (0 or 1), filter by priority label
        min_score: Minimum similarity score threshold (0-1)

    Returns:
        dict with keys:
            - results: list of SearchResult dicts
            - query: original query
            - latency_ms: retrieval time in milliseconds
            - total_found: number of results above threshold
            - low_similarity: True if best result below threshold

    Raises:
        ValueError: If query is empty
        RuntimeError: If Qdrant search fails
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1 or top_k > 50:
        raise ValueError("top_k must be between 1 and 50")

    logger.info(
        "Retrieving top-%d results for query: '%s...'",
        top_k, query[:50]
    )

    start_time = time.time()

    try:
        # Encode query to vector
        query_vector = encode_text(query)

        # Build optional priority filter
        search_filter = None
        if priority_filter is not None:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="priority",
                        match=MatchValue(value=priority_filter)
                    )
                ]
            )

        # Search Qdrant
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
            score_threshold=min_score
        )

        # Parse results
        results = []
        for hit in search_results.points:
            result = SearchResult(
                text=hit.payload.get("text", ""),
                cleaned_text=hit.payload.get("cleaned_text", ""),
                priority=hit.payload.get("priority", 0),
                score=hit.score,
                tweet_id=hit.payload.get("tweet_id", 0)
            )
            results.append(result.to_dict())

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Check if results are meaningful
        low_similarity = (
            len(results) == 0 or
            results[0]["similarity_score"] < SIMILARITY_THRESHOLD
        )

        if low_similarity:
            logger.warning(
                "Low similarity results for query: '%s...'", query[:50]
            )

        response = {
            "results": results,
            "query": query,
            "latency_ms": latency_ms,
            "total_found": len(results),
            "low_similarity": low_similarity
        }

        logger.info(
            "Retrieved %d results in %s ms", len(results), latency_ms
        )
        return response

    except Exception as e:
        logger.error("Retrieval failed for query '%s': %s", query[:50], e)
        raise RuntimeError(f"Retrieval failed: {e}") from e


# ── Context Builder ───────────────────────────────────────────────────────────
def build_rag_context(results: list[dict], max_tickets: int = 5) -> str:
    """
    Format retrieved tickets as context for the LLM prompt.

    Args:
        results: List of SearchResult dicts from retrieve()
        max_tickets: Maximum number of tickets to include

    Returns:
        Formatted string ready to inject into LLM prompt
    """
    if not results:
        return "No similar past tickets found."

    context_parts = []
    for i, result in enumerate(results[:max_tickets], 1):
        priority_label = "URGENT" if result["priority"] == 1 else "NORMAL"
        similarity = result["similarity_score"]
        context_parts.append(
            f"[Ticket {i}] "
            f"Priority: {priority_label} | "
            f"Similarity: {similarity:.2f}\n"
            f"{result['text']}"
        )

    return "\n\n".join(context_parts)