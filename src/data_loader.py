"""
Data Loader Module for E-Waste Policy Pulse.
Handles CSV ingestion, column normalisation, schema validation,
deduplication, blank record removal, and reporting of ingestion statistics.
"""

from io import BytesIO, StringIO
from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names to canonical names:
    'post_id', 'text', 'date', 'platform', 'likes'.
    """
    column_mapping = {}
    lower_cols = {col.lower().strip(): col for col in df.columns}

    # Text mapping (mandatory)
    text_aliases = ["text", "post_text", "content", "tweet", "post", "message", "body"]
    for alias in text_aliases:
        if alias in lower_cols:
            column_mapping[lower_cols[alias]] = "text"
            break

    # ID mapping
    id_aliases = ["post_id", "id", "tweet_id", "status_id", "record_id"]
    for alias in id_aliases:
        if alias in lower_cols:
            column_mapping[lower_cols[alias]] = "post_id"
            break

    # Date mapping
    date_aliases = ["date", "created_at", "timestamp", "post_date", "time"]
    for alias in date_aliases:
        if alias in lower_cols:
            column_mapping[lower_cols[alias]] = "date"
            break

    # Platform mapping
    platform_aliases = ["platform", "source", "network", "channel", "social_media"]
    for alias in platform_aliases:
        if alias in lower_cols:
            column_mapping[lower_cols[alias]] = "platform"
            break

    # Engagement mapping
    likes_aliases = ["likes", "engagement_count", "like_count", "favorites", "retweets", "engagement"]
    for alias in likes_aliases:
        if alias in lower_cols:
            column_mapping[lower_cols[alias]] = "likes"
            break

    df = df.rename(columns=column_mapping)
    return df


def load_dataset(
    file_or_path: Union[str, BytesIO, StringIO, Any],
    is_demonstration: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load, validate and clean CSV dataset.

    Parameters:
    -----------
    file_or_path : str or file-like object
        File path or Streamlit UploadedFile buffer.
    is_demonstration : bool
        Flag indicating whether this is the provided sample demonstration dataset.

    Returns:
    --------
    cleaned_df : pd.DataFrame
        DataFrame with canonical columns and valid text records.
    stats : dict
        Summary of ingestion (raw count, dropped duplicates, blank removed, retained).
    """
    # 1. Read CSV
    try:
        df = pd.read_csv(file_or_path)
    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file is completely empty. Please upload a valid CSV file.")
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file: {str(exc)}")

    if df.empty:
        raise ValueError("The provided CSV file contains no data rows.")

    raw_count = len(df)

    # 2. Normalise column names
    df = normalize_column_names(df)

    # 3. Check for mandatory 'text' column
    if "text" not in df.columns:
        cols_found = ", ".join(f"'{c}'" for c in df.columns)
        raise ValueError(
            f"Mandatory 'text' column was not found. Detected columns: [{cols_found}]. "
            "Please ensure your CSV contains a column named 'text' or 'content'."
        )

    # 4. Handle post_id
    if "post_id" not in df.columns:
        df["post_id"] = [f"POST_{i+1:04d}" for i in range(len(df))]
    else:
        df["post_id"] = df["post_id"].astype(str)

    # 5. Handle empty or blank text records
    # Convert text to string, strip whitespace
    df["text"] = df["text"].astype(str).str.strip()
    # Filter out empty strings, 'nan', 'none', 'null'
    non_blank_mask = (
        (df["text"].str.len() > 0) &
        (~df["text"].str.lower().isin(["nan", "none", "null", "undefined"]))
    )
    blank_dropped = int((~non_blank_mask).sum())
    df = df[non_blank_mask].copy()

    # 6. Deduplication based on cleaned text string
    # Remove exact duplicate texts (case-insensitive)
    text_lower = df["text"].str.lower()
    duplicate_mask = text_lower.duplicated(keep="first")
    duplicates_dropped = int(duplicate_mask.sum())
    df = df[~duplicate_mask].copy()

    # 7. Handle optional 'date' column
    has_date = "date" in df.columns
    if has_date:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        # Keep original date string as well
        df["date"] = df["date"].astype(str)
    else:
        df["date"] = "Unknown"
        df["date_parsed"] = pd.NaT

    # 8. Handle optional 'platform' column
    has_platform = "platform" in df.columns
    if has_platform:
        df["platform"] = df["platform"].fillna("Unknown").astype(str).str.strip()
        # Capitalize nicely if standard
        df["platform"] = df["platform"].replace({"twitter": "Twitter", "linkedin": "LinkedIn", "reddit": "Reddit", "facebook": "Facebook", "instagram": "Instagram"})
    else:
        df["platform"] = "General Social"

    # 9. Handle optional 'likes' / engagement count
    has_engagement = "likes" in df.columns
    if has_engagement:
        df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    else:
        df["likes"] = 0

    retained_count = len(df)
    df = df.reset_index(drop=True)

    stats = {
        "raw_count": raw_count,
        "blank_dropped": blank_dropped,
        "duplicates_dropped": duplicates_dropped,
        "retained_count": retained_count,
        "has_date": has_date,
        "has_platform": has_platform,
        "has_engagement": has_engagement,
        "is_demonstration": is_demonstration,
    }

    return df, stats
