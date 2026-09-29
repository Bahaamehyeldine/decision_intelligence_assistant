"""Hand-crafted urgency features: the same logic as the training notebook."""

import numpy as np
import pytest


def test_clean_text_strips_urls_mentions_hashtags(features_module):
    raw = "@AmazonHelp my order https://t.co/x is LATE &amp; broken #fail"
    assert features_module.clean_text(raw) == "my order is late & broken"


@pytest.mark.parametrize(
    "text, keyword, negated",
    [
        ("the app is not broken anymore", "broken", True),
        ("the app is broken again", "broken", False),
        ("never had an error before today", "error", True),
    ],
)
def test_is_negated(features_module, text, keyword, negated):
    assert features_module.is_negated(text, keyword) is negated


def test_extract_features_shape_and_signals(features_module):
    feats = features_module.extract_features("I was DOUBLE CHARGED!!! refund me now, no service for days?")
    assert feats.shape == (1, 113)
    hand = feats[0, :13]
    assert hand[1] == 1          # financial keyword
    assert hand[2] == 3          # exclamation marks
    assert hand[4] >= 2          # ALL-CAPS words
    assert hand[8] == 1          # question marks
    assert hand[10] == 1         # service-down keyword
    assert np.isfinite(feats).all()


def test_extract_features_rejects_empty(features_module):
    with pytest.raises(ValueError):
        features_module.extract_features("   ")
