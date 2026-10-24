"""
Unit tests for Sentiment Analysis module.
Validates polarity thresholds, score bounds, and metric aggregations.
"""

import pandas as pd
import pytest
from src.sentiment_analysis import (
    compute_sentiment_single,
    analyze_dataframe_sentiment,
    get_sentiment_metrics,
    POLARITY_THRESHOLD_POSITIVE,
    POLARITY_THRESHOLD_NEGATIVE
)


def test_positive_classification():
    text = "Great job by the municipal corporation on the efficient e-waste recycling centre!"
    pol, subj, label = compute_sentiment_single(text)
    assert pol > POLARITY_THRESHOLD_POSITIVE
    assert label == "Positive"
    assert 0.0 <= subj <= 1.0


def test_negative_classification():
    text = "Horrible toxic pollution and illegal burning of electronics destroying local health."
    pol, subj, label = compute_sentiment_single(text)
    assert pol < POLARITY_THRESHOLD_NEGATIVE
    assert label == "Negative"
    assert 0.0 <= subj <= 1.0


def test_neutral_classification():
    text = "CPCB notification regarding e-waste quarterly return filing date for manufacturers."
    pol, subj, label = compute_sentiment_single(text)
    # Neutral text should have polarity == 0.0
    assert pol == 0.0
    assert label == "Neutral"


def test_empty_text_returns_neutral():
    pol, subj, label = compute_sentiment_single("")
    assert pol == 0.0
    assert subj == 0.0
    assert label == "Neutral"


def test_dataframe_metrics_sum_consistency():
    sample_data = {
        "text": [
            "Excellent recycling initiative for smartphones!",
            "Terrible illegal dumping of heavy metals.",
            "Meeting scheduled for e-waste policy review.",
            "Another formal collection kiosk launched today."
        ]
    }
    df = pd.DataFrame(sample_data)
    analyzed_df = analyze_dataframe_sentiment(df, text_column="text")
    metrics = get_sentiment_metrics(analyzed_df)

    # Validate that counts add up exactly to total
    assert metrics["positive_count"] + metrics["negative_count"] + metrics["neutral_count"] == metrics["total_count"]
    assert metrics["total_count"] == len(df)
    # Validate percentage sums approximately 100%
    total_pct = metrics["positive_pct"] + metrics["negative_pct"] + metrics["neutral_pct"]
    assert pytest.approx(total_pct, rel=1e-2) == 100.0


def test_score_boundaries():
    texts = [
        "Best recycling in the world!",
        "Worst catastrophe ever.",
        "Neutral notice document."
    ]
    for t in texts:
        pol, subj, _ = compute_sentiment_single(t)
        assert -1.0 <= pol <= 1.0
        assert 0.0 <= subj <= 1.0
