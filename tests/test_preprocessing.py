"""
Unit tests for NLP preprocessing module.
Validates text cleaning, negation preservation, lemmatisation, and edge cases.
"""

import pytest
from src.preprocessing import (
    clean_text_structure,
    preprocess_post,
    PRESERVED_NEGATIONS
)


def test_clean_text_structure_removes_urls_and_html():
    raw = "Great e-waste drive! Visit <a href='http://recycle.in'>http://recycle.in</a> or www.cpcb.gov.in &amp; join us!"
    cleaned = clean_text_structure(raw)
    assert "http" not in cleaned
    assert "www" not in cleaned
    assert "<a" not in cleaned
    assert "&amp;" not in cleaned
    assert "e-waste" in cleaned


def test_clean_text_structure_removes_handles_keeps_hashtags():
    raw = "Hey @moefcc check out the new recycling center #CircularEconomy #EWaste"
    cleaned = clean_text_structure(raw)
    assert "@moefcc" not in cleaned
    assert "CircularEconomy" in cleaned
    assert "#" not in cleaned


def test_negations_are_preserved():
    # If negations were stripped as standard stopwords, "not working" becomes "working",
    # flipping sentiment from negative to positive.
    text = "The e-waste drop-off bin is not working and offers no assistance."
    res = preprocess_post(text, lemmatize=False, remove_stopwords=True)
    tokens = res["tokens"]

    assert "not" in tokens, f"'not' should be preserved in tokens. Found: {tokens}"
    assert "no" in tokens, f"'no' should be preserved in tokens. Found: {tokens}"


def test_lemmatization():
    text = "Recyclers are processing discarded electronic batteries and laptops."
    res = preprocess_post(text, lemmatize=True, remove_stopwords=True)
    tokens = res["tokens"]

    # "batteries" should be lemmatized to "battery"
    assert "battery" in tokens
    # "laptops" should be lemmatized to "laptop"
    assert "laptop" in tokens


def test_empty_and_whitespace_text_handling():
    res1 = preprocess_post("", lemmatize=True, remove_stopwords=True)
    assert res1["cleaned_text"] == ""
    assert res1["tokens"] == []

    res2 = preprocess_post("    \n\t   ", lemmatize=True, remove_stopwords=True)
    assert res2["cleaned_text"] == ""
    assert res2["tokens"] == []


def test_canonical_ewaste_term():
    raw1 = "We need better e waste collection"
    raw2 = "We need better ewaste collection"
    res1 = preprocess_post(raw1)
    res2 = preprocess_post(raw2)
    assert "e-waste" in res1["tokens"]
    assert "e-waste" in res2["tokens"]
