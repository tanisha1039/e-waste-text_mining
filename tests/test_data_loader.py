"""
Unit tests for Data Loader module.
Validates CSV ingestion, column normalisation, blank row dropping,
deduplication, and ingestion statistics.
"""

from io import StringIO
import pytest
from src.data_loader import load_dataset, normalize_column_names
import pandas as pd


def test_load_sample_csv_success():
    csv_path = "data/sample_posts.csv"
    df, stats = load_dataset(csv_path, is_demonstration=True)

    assert not df.empty
    assert "text" in df.columns
    assert "post_id" in df.columns
    assert stats["raw_count"] > 80
    assert stats["duplicates_dropped"] >= 1
    assert stats["blank_dropped"] >= 1
    assert stats["retained_count"] == len(df)
    assert stats["raw_count"] == stats["retained_count"] + stats["duplicates_dropped"] + stats["blank_dropped"]


def test_column_normalisation_aliases():
    raw_df = pd.DataFrame({
        "status_id": [1],
        "content": ["Test post on e-waste rules"],
        "created_at": ["2024-01-01"],
        "source": ["Twitter"],
        "favorites": [10]
    })
    normalized = normalize_column_names(raw_df)
    assert "post_id" in normalized.columns
    assert "text" in normalized.columns
    assert "date" in normalized.columns
    assert "platform" in normalized.columns
    assert "likes" in normalized.columns


def test_empty_csv_raises_value_error():
    empty_csv = StringIO("")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(empty_csv)


def test_missing_text_column_raises_value_error():
    invalid_csv = StringIO("id,date,platform\n1,2024-01-01,Twitter\n")
    with pytest.raises(ValueError, match="Mandatory 'text' column was not found"):
        load_dataset(invalid_csv)


def test_deduplication_and_blank_removal():
    test_csv = StringIO(
        "text,platform\n"
        "Great e-waste drive,Twitter\n"
        "Great e-waste drive,Twitter\n"
        "   ,Facebook\n"
        "nan,Reddit\n"
        "Unique e-waste post,LinkedIn\n"
    )
    df, stats = load_dataset(test_csv)
    assert len(df) == 2
    assert stats["raw_count"] == 5
    assert stats["duplicates_dropped"] == 1
    assert stats["blank_dropped"] == 2
    assert stats["retained_count"] == 2
