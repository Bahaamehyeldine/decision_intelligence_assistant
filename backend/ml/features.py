"""
features.py
-----------
Feature extraction for ML priority classification.
Extracts the same 13 hand-crafted + 100 TF-IDF/SVD features
used during model training.

CRITICAL: Uses pre-fitted TF-IDF and SVD from training.
Never refit on new data — would cause training-serving skew.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Optional

import emoji
import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── Load Pre-fitted Transformers ──────────────────────────────────────────────
try:
    tfidf = joblib.load(os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl"))
    svd = joblib.load(os.path.join(PROCESSED_DIR, "svd_reducer.pkl"))
    normalizer = joblib.load(os.path.join(PROCESSED_DIR, "normalizer.pkl"))
    logger.info("Loaded pre-fitted TF-IDF, SVD, and normalizer")
except Exception as e:
    logger.error("Failed to load transformers: %s", e)
    raise RuntimeError(f"Cannot load feature transformers: {e}") from e

# ── NLP Setup ─────────────────────────────────────────────────────────────────
vader = SentimentIntensityAnalyzer()

# ── Keyword Sets ──────────────────────────────────────────────────────────────
PROBLEM_KEYWORDS = {
    "not working", "broken", "failed", "fail", "error", "issue",
    "problem", "bug", "crash", "crashed", "outage", "offline",
    "unavailable", "unable", "cannot", "can't", "wont", "won't",
    "locked", "blocked", "banned", "suspended", "lost access",
    "hacked", "compromised", "not loading", "freezing", "frozen",
    "stuck", "terrible", "awful", "horrible", "worst", "unacceptable",
    "ridiculous", "disgusting", "useless", "pathetic", "disappointed",
    "frustrated", "angry", "furious", "fed up", "sick of", "tired of"
}
FINANCIAL_KEYWORDS = {
    "refund", "charged", "overcharged", "stolen", "fraud", "scam",
    "unauthorized", "double charged", "wrong charge"
}
SERVICE_DOWN_KEYWORDS = {
    "no service", "no signal", "no internet", "no connection",
    "disconnected", "complete outage", "totally down", "not available"
}
URGENT_EMOJIS = {
    ":enraged_face:", ":angry_face:", ":face_with_symbols_on_mouth:",
    ":crying_face:", ":loudly_crying_face:", ":broken_heart:",
    ":fire:", ":sos:", ":warning:", ":red_circle:", ":skull:",
    ":face_screaming_in_fear:", ":weary_face:", ":tired_face:",
    ":disappointed_face:", ":thumbs_down:"
}
NEGATIONS = {
    "not", "never", "no", "fixed", "resolved", "working now",
    "don't", "didn't", "doesn't", "wasn't", "weren't", "won't"
}


# ── Helper Functions ──────────────────────────────────────────────────────────
def is_negated(text: str, keyword: str) -> bool:
    """Check if keyword is preceded by negation within 3 words."""
    words = text.split()
    for i, word in enumerate(words):
        if keyword in word:
            window = words[max(0, i-3):i]
            if any(neg in window for neg in NEGATIONS):
                return True
    return False


def clean_text(text: str) -> str:
    """Clean text for feature extraction."""
    import html
    text = html.unescape(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def prepare_for_tfidf(text: str) -> str:
    """Prepare text for TF-IDF — same pipeline as training."""
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    import nltk

    lemmatizer = WordNetLemmatizer()

    COMPANY_STOPWORDS = {
        'amazon', 'apple', 'google', 'twitter', 'uber', 'netflix',
        'spotify', 'att', 'verizon', 'comcast', 'tmobile', 'sprint'
    }

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN,
                    'V': wordnet.VERB, 'R': wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    text = re.sub(r'\b\d+(?:st|nd|rd|th)?\b', '', text)
    words = text.split()
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word))
        for word in words
        if word not in COMPANY_STOPWORDS and len(word) > 2
    ]
    return ' '.join(lemmatized)


# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(text: str) -> np.ndarray:
    """
    Extract 113 features from raw text for ML prediction.

    Applies same pipeline as training:
    - 13 hand-crafted features
    - TF-IDF transform (pre-fitted, 500 terms)
    - SVD reduction (pre-fitted, 100 components)
    - Normalization

    Args:
        text: Raw user query text

    Returns:
        numpy array of shape (1, 113) ready for model.predict()

    Raises:
        ValueError: If text is empty
        RuntimeError: If feature extraction fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    try:
        cleaned = clean_text(text)
        vader_scores = vader.polarity_scores(cleaned)
        emoji_text = emoji.demojize(text)
        tfidf_text = prepare_for_tfidf(cleaned)

        # ── Hand-crafted features (13) ────────────────────────────────────
        hand_crafted = np.array([[
            # 1. Problem keyword count (with negation check)
            sum(1 for kw in PROBLEM_KEYWORDS
                if kw in cleaned and not is_negated(cleaned, kw)),
            # 2. Financial keyword
            int(any(kw in cleaned for kw in FINANCIAL_KEYWORDS)),
            # 3. Exclamation marks
            text.count('!'),
            # 4. Word repetition
            int(bool(re.search(
                r'\b(\w+)(?:\s+\1){2,}\b', cleaned, re.IGNORECASE))),
            # 5. ALL CAPS words
            len(re.findall(r'\b[A-Z]{3,}\b', text)),
            # 6. VADER compound sentiment
            vader_scores['compound'],
            # 7. VADER negative score
            vader_scores['neg'],
            # 8. VADER positive score
            vader_scores['pos'],
            # 9. Question marks
            text.count('?'),
            # 10. Text length
            len(cleaned.split()),
            # 11. Service down keyword
            int(any(kw in cleaned for kw in SERVICE_DOWN_KEYWORDS)),
            # 12. Urgent emoji
            int(any(e in emoji_text for e in URGENT_EMOJIS)),
            # 13. RoBERTa negative (use VADER neg as proxy at inference)
            vader_scores['neg']
        ]], dtype=np.float64)

        # ── TF-IDF + SVD features (100) ───────────────────────────────────
        tfidf_vec = tfidf.transform([tfidf_text])
        svd_vec = svd.transform(tfidf_vec)
        svd_normalized = normalizer.transform(svd_vec)

        # ── Combine (113 total) ───────────────────────────────────────────
        features = np.hstack([hand_crafted, svd_normalized])

        logger.debug("Extracted features shape: %s", features.shape)
        return features

    except Exception as e:
        logger.error("Feature extraction failed: %s", e)
        raise RuntimeError(f"Feature extraction failed: {e}") from e