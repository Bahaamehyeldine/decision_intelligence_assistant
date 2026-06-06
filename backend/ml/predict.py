"""
predict.py
----------
ML priority classification for support tickets.
Loads trained Gradient Boosting model and predicts urgent/normal.
"""

from __future__ import annotations

import logging
import os
import time

import joblib
import numpy as np

from ml.features import extract_features

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Model Loading ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "priority_classifier.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded priority classifier from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    raise RuntimeError(f"Cannot load priority classifier: {e}") from e


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_priority(text: str) -> dict:
    """
    Predict ticket priority using trained ML model.

    Args:
        text: Raw user query text

    Returns:
        dict with keys:
            - priority: 0 (normal) or 1 (urgent)
            - priority_label: "normal" or "urgent"
            - confidence: probability of predicted class (0-1)
            - urgent_probability: probability of urgent class
            - latency_ms: prediction time in milliseconds
            - model: model type name
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    logger.info("Predicting priority for: '%s...'", text[:50])
    start = time.time()

    try:
        # Extract features
        features = extract_features(text)

        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        urgent_prob = float(probabilities[1])
        confidence = float(max(probabilities))
        latency_ms = round((time.time() - start) * 1000, 2)

        result = {
            "priority": int(prediction),
            "priority_label": "urgent" if prediction == 1 else "normal",
            "confidence": round(confidence, 4),
            "urgent_probability": round(urgent_prob, 4),
            "latency_ms": latency_ms,
            "model": type(model).__name__
        }

        logger.info(
            "Prediction: %s (confidence: %.2f) in %sms",
            result["priority_label"], confidence, latency_ms
        )
        return result

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise RuntimeError(f"Priority prediction failed: {e}") from e