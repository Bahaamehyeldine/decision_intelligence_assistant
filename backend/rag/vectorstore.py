"""
vectorstore.py
--------------
Qdrant vector store management for the RAG system.
Handles collection creation, tweet indexing, and similarity search.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.embeddings import encode_batch, EMBEDDING_DIM

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
COLLECTION_NAME = "support_tickets"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
UPLOAD_BATCH_SIZE = 500


# ── Custom Exceptions ─────────────────────────────────────────────────────────
class QdrantConnectionError(Exception):
    """Raised when Qdrant connection cannot be established."""
    pass


class CollectionError(Exception):
    """Raised when collection operations fail."""
    pass


# ── Client Setup ──────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def _create_client() -> QdrantClient:
    """
    Create Qdrant client with retry logic.
    Retries 3 times with exponential backoff if connection fails.
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()  # verify connection
        logger.info("Connected to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
        return client
    except Exception as e:
        logger.error("Qdrant connection failed: %s", e)
        raise QdrantConnectionError(
            f"Cannot connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}"
        ) from e


client = _create_client()


# ── Collection Management ─────────────────────────────────────────────────────
def create_collection(recreate: bool = False) -> None:
    """
    Create Qdrant collection for support tickets.

    Args:
        recreate: If True, delete existing collection and recreate.
                  Use carefully — deletes all indexed data.

    Raises:
        CollectionError: If collection creation fails.
    """
    try:
        existing = [c.name for c in client.get_collections().collections]

        if COLLECTION_NAME in existing:
            if recreate:
                client.delete_collection(COLLECTION_NAME)
                logger.warning(
                    "Deleted existing collection: %s", COLLECTION_NAME
                )
            else:
                logger.info(
                    "Collection already exists: %s", COLLECTION_NAME
                )
                return

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        logger.info("Created collection: %s", COLLECTION_NAME)

    except Exception as e:
        raise CollectionError(f"Failed to create collection: {e}") from e


# ── Indexing ──────────────────────────────────────────────────────────────────
def index_tweets(
    df: pd.DataFrame,
    batch_size: int = UPLOAD_BATCH_SIZE
) -> dict:
    """
    Generate embeddings for all tweets and store in Qdrant.

    Args:
        df: DataFrame with columns [tweet_id, text, cleaned_text, priority]
        batch_size: Number of points to upload per Qdrant batch

    Returns:
        dict with keys: total_indexed, failed, duration_seconds

    Raises:
        ValueError: If required columns are missing
        CollectionError: If upload fails
    """
    # Input validation
    required_cols = ['tweet_id', 'text', 'cleaned_text', 'priority']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Starting indexing of %d tweets", len(df))
    start_time = time.time()
    failed = 0

    # Check if already indexed
    info = get_collection_info()
    if info and info.get('total_vectors', 0) >= len(df):
        logger.info(
            "Collection already has %d vectors — skipping indexing",
            info['total_vectors']
        )
        return {
            "total_indexed": info['total_vectors'],
            "failed": 0,
            "duration_seconds": 0,
            "skipped": True
        }

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = encode_batch(df['cleaned_text'].tolist())

    # Upload in batches
    logger.info("Uploading to Qdrant in batches of %d...", batch_size)
    points = []

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            point = PointStruct(
                id=int(row['tweet_id']) % (2**31),
                vector=embeddings[i].tolist(),
                payload={
                    "text": str(row['text']),
                    "cleaned_text": str(row['cleaned_text']),
                    "priority": int(row['priority']),
                    "tweet_id": int(row['tweet_id'])
                }
            )
            points.append(point)

            if len(points) == batch_size:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                logger.info(
                    "Uploaded %d/%d tweets", i + 1, len(df)
                )
                points = []

        except Exception as e:
            logger.error("Failed to create point for tweet %s: %s",
                        row['tweet_id'], e)
            failed += 1

    # Upload remaining
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    elapsed = time.time() - start_time
    result = {
        "total_indexed": len(df) - failed,
        "failed": failed,
        "duration_seconds": round(elapsed, 2)
    }
    logger.info("Indexing complete: %s", result)
    return result


# ── Collection Info ───────────────────────────────────────────────────────────
def get_collection_info() -> Optional[dict]:
    """
    Get information about the support tickets collection.

    Returns:
        dict with collection stats, or None if collection doesn't exist
    """
    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "total_vectors": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance)
        }
    except Exception:
        return None


# ── Health Check ──────────────────────────────────────────────────────────────
def health_check() -> dict:
    """
    Verify Qdrant connection and collection status.

    Returns:
        dict with status and collection info
    """
    try:
        client.get_collections()
        info = get_collection_info()
        return {
            "status": "healthy",
            "qdrant_host": QDRANT_HOST,
            "qdrant_port": QDRANT_PORT,
            "collection": info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
    