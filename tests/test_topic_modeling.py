"""
Unit tests for Topic Modelling module.
Validates LDA topic extraction, topic distribution, and small-data guards.
"""

import pandas as pd
import pytest
from src.preprocessing import preprocess_dataframe
from src.topic_modeling import perform_lda_topic_modeling, GENSIM_AVAILABLE


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim is not installed")
def test_lda_topic_modeling_success():
    # Create synthetic tokenised dataset
    docs = [
        "e-waste recycling centre municipal corporation drop-off drive electronics",
        "toxic heavy metals lead mercury poisoning soil water informal dismantler",
        "extended producer responsibility epr portal registration cpcb guidelines compliance",
        "smartphone battery refurbishment circular economy repair components gadgets",
        "informal scrap workers health risk burning wire open fire safety gear",
        "brand take-back scheme failure drop-off bin missing citizen awareness",
        "cpcb notice quarterly returns electronic equipment recycling targets",
        "lithium-ion battery storage fire hazard scrap yard regulation",
        "college awareness campaign students electronic waste segregation",
        "hydrometallurgical plant recovery precious gold copper sustainable tech",
        "deposit refund system kiosks bus pass discount citizen participation",
        "illegal dumping broken appliances river floodplain show cause notice"
    ]
    df = pd.DataFrame({"text": docs})
    df = preprocess_dataframe(df)

    res = perform_lda_topic_modeling(df, num_topics=3, passes=5)
    assert res["success"] is True
    assert res["num_topics"] == 3
    assert len(res["topics"]) == 3
    assert "dominant_topic" in res["dominant_df"].columns
    assert len(res["dominant_df"]) == len(df)


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim is not installed")
def test_lda_insufficient_data_guard():
    df = pd.DataFrame({"text": ["Too few posts", "Not enough tokens"]})
    df = preprocess_dataframe(df)
    res = perform_lda_topic_modeling(df, num_topics=3)
    assert res["success"] is False
    assert "minimum token requirements" in res["message"] or "requires at least" in res["message"]
