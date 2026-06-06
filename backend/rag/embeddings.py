"""
embeddings.py
-------------
Sentence embedding generation for the RAG system.
Uses sentence-transformers with GPU acceleration.

Author: Decision Intelligence Assistant
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64

# ── Device Setup ──────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Embedding device: %s", device)

# ── Model Loading ─────────────────────────────────────────────────────────────
try:
    _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
except Exception as e:
    logger.error("Failed to load embedding model: %s", e)
    raise RuntimeError(f"Cannot load embedding model {EMBEDDING_MODEL}: {e}") from e


# ── Text Normalization ────────────────────────────────────────────────────────
def normalize_for_embedding(text: str) -> str:
    """
    Normalize text before embedding to improve similarity matching.
    
    Removes URLs, mentions, hashtags and extra whitespace that would
    distort semantic similarity between otherwise identical messages.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned, lowercased text ready for embedding
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Single Text Encoding ──────────────────────────────────────────────────────
@lru_cache(maxsize=1000)
def _encode_cached(text: str) -> tuple[float, ...]:
    """
    Internal cached encoding function.
    Cache stores last 1000 unique queries for instant retrieval.
    
    Args:
        text: Normalized text string (must be hashable)
        
    Returns:
        Embedding as tuple of floats (hashable for cache)
    """
    try:
        if not text or not text.strip():
            logger.warning("Empty text received for embedding")
            return tuple([0.0] * EMBEDDING_DIM)
        embedding = _model.encode(text, convert_to_numpy=True)
        return tuple(float(x) for x in embedding)
    except Exception as e:
        logger.error("Embedding error for text '%s...': %s", text[:50], e)
        return tuple([0.0] * EMBEDDING_DIM)


def encode_text(text: str) -> list[float]:
    """
    Encode a single text string into an embedding vector.
    
    Used for encoding user queries at retrieval time.
    Results are cached — identical queries return instantly.
    
    Args:
        text: Input text (raw, normalization applied internally)
        
    Returns:
        Embedding vector as list of floats, length EMBEDDING_DIM (384)
        Returns zero vector if text is empty or encoding fails.
        
    Example:
        >>> vec = encode_text("my internet is not working")
        >>> len(vec)
        384
    """
    normalized = normalize_for_embedding(text)
    return list(_encode_cached(normalized))


# ── Batch Encoding ────────────────────────────────────────────────────────────
def encode_batch(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True
) -> np.ndarray:
    """
    GPU-accelerated batch encoding for offline indexing.
    
    Processes texts in batches for memory efficiency.
    Used once to embed all tweets before storing in Qdrant.
    
    Args:
        texts: List of raw text strings
        batch_size: GPU batch size. Increase if VRAM allows (RTX 5070: try 128)
        show_progress: Show tqdm progress bar
        
    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIM)
        Zero rows inserted for any failed encodings.
        
    Raises:
        ValueError: If texts list is empty
    """
    if not texts:
        raise ValueError("Cannot encode empty text list")

    logger.info(
        "Encoding %d texts in batches of %d on %s",
        len(texts), batch_size, device
    )

    try:
        normalized = [normalize_for_embedding(t) for t in texts]
        embeddings = _model.encode(
            normalized,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=device,
            normalize_embeddings=True
        )
        logger.info(
            "Batch encoding complete. Shape: %s", embeddings.shape
        )
        return embeddings

    except Exception as e:
        logger.error("Batch encoding failed: %s", e)
        return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)


# ── Utility ───────────────────────────────────────────────────────────────────
def get_model_info() -> dict:
    """Return embedding model metadata."""
    return {
        "model_name": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "device": device,
        "cache_info": _encode_cached.cache_info()._asdict()
    }