"""
Sentiment Analysis Module for E-Waste Policy Pulse.
Calculates TextBlob sentiment polarity and subjectivity.
Applies strictly defined classification rules and provides statistical summaries.
"""

from typing import Any, Dict, Tuple
import pandas as pd
from textblob import TextBlob


# Explicit classification thresholds
POLARITY_THRESHOLD_POSITIVE = 0.0
POLARITY_THRESHOLD_NEGATIVE = 0.0

CLASSIFICATION_RULES_EXPLANATION = """
Classification Rule Thresholds:
- Positive: Polarity > 0.0 (Expresses approval, satisfaction, or favorable sentiment)
- Negative: Polarity < 0.0 (Expresses dissatisfaction, criticism, alarm, or concern)
- Neutral:  Polarity == 0.0 (Objective statements, official notices, policy facts, or factual inquiries)
"""

TEXTBLOB_ACADEMIC_DISCLAIMER = """
Academic Evaluation & Methodological Disclaimer:
TextBlob provides a lexicon-based and pattern-matching baseline for sentiment analysis.
It assigns polarity [-1.0 to +1.0] and subjectivity [0.0 to 1.0] by looking up adjectives
and adverbs in the Pattern dictionary.
Important Considerations:
1. Sarcasm & Irony: Baseline lexicon models frequently misclassify sarcastic posts.
2. Domain-Specific Terminology: Words like 'hazard', 'toxic', 'waste' or 'acid' often register
   as negative even when used in objective policy or engineering contexts (e.g., 'handling hazardous waste safely').
3. Context Blindness: Rule-based models analyze isolated lexical tokens and do not possess
   the contextual semantics of modern transformer-based models (like BERT).
Hence, TextBlob is an interpretable baseline, not a domain-fine-tuned classifier.
"""


def compute_sentiment_single(text: str) -> Tuple[float, float, str]:
    """
    Compute polarity, subjectivity, and sentiment label for a single text.

    Returns:
    --------
    (polarity, subjectivity, label)
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0, "Neutral"

    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    if polarity > POLARITY_THRESHOLD_POSITIVE:
        label = "Positive"
    elif polarity < POLARITY_THRESHOLD_NEGATIVE:
        label = "Negative"
    else:
        label = "Neutral"

    return polarity, subjectivity, label


def analyze_dataframe_sentiment(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Apply sentiment analysis across the DataFrame on the specified text column
    (uses the original text to preserve full linguistic nuances and modifiers).

    Adds:
    - 'polarity': float [-1.0, 1.0]
    - 'subjectivity': float [0.0, 1.0]
    - 'sentiment_label': 'Positive' | 'Negative' | 'Neutral'
    """
    polarities = []
    subjectivities = []
    labels = []

    for text in df[text_column]:
        pol, subj, lbl = compute_sentiment_single(text)
        polarities.append(pol)
        subjectivities.append(subj)
        labels.append(lbl)

    df["polarity"] = polarities
    df["subjectivity"] = subjectivities
    df["sentiment_label"] = labels

    return df


def get_sentiment_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive sentiment distribution metrics.

    Returns:
    --------
    dict containing:
        - total_count: int
        - positive_count: int
        - negative_count: int
        - neutral_count: int
        - positive_pct: float
        - negative_pct: float
        - neutral_pct: float
        - avg_polarity: float
        - avg_subjectivity: float
    """
    total = len(df)
    if total == 0:
        return {
            "total_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "avg_polarity": 0.0,
            "avg_subjectivity": 0.0,
        }

    counts = df["sentiment_label"].value_counts().to_dict()
    pos_count = counts.get("Positive", 0)
    neg_count = counts.get("Negative", 0)
    neu_count = counts.get("Neutral", 0)

    # Verification: pos + neg + neu must equal total
    assert pos_count + neg_count + neu_count == total, "Sentiment counts do not sum to total posts"

    metrics = {
        "total_count": total,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "positive_pct": round((pos_count / total) * 100, 2),
        "negative_pct": round((neg_count / total) * 100, 2),
        "neutral_pct": round((neu_count / total) * 100, 2),
        "avg_polarity": round(float(df["polarity"].mean()), 4),
        "avg_subjectivity": round(float(df["subjectivity"].mean()), 4),
    }

    return metrics


def filter_posts(
    df: pd.DataFrame,
    sentiment_filter: str = "All",
    platform_filter: str = "All",
    start_date=None,
    end_date=None,
    search_query: str = ""
) -> pd.DataFrame:
    """
    Filter DataFrame by sentiment, platform, date range, and search keyword.
    """
    filtered = df.copy()

    # Sentiment filter
    if sentiment_filter != "All" and "sentiment_label" in filtered.columns:
        filtered = filtered[filtered["sentiment_label"] == sentiment_filter]

    # Platform filter
    if platform_filter != "All" and "platform" in filtered.columns:
        filtered = filtered[filtered["platform"] == platform_filter]

    # Date range filter (if parsed dates are available)
    if "date_parsed" in filtered.columns and start_date and end_date:
        valid_dates_mask = filtered["date_parsed"].notna()
        date_mask = (
            (filtered["date_parsed"].dt.date >= start_date) &
            (filtered["date_parsed"].dt.date <= end_date)
        )
        filtered = filtered[valid_dates_mask & date_mask]

    # Search keyword filter
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        search_mask = (
            filtered["text"].str.lower().str.contains(q, regex=False, na=False) |
            (filtered["cleaned_text"].str.lower().str.contains(q, regex=False, na=False) if "cleaned_text" in filtered.columns else False)
        )
        filtered = filtered[search_mask]

    return filtered.reset_index(drop=True)
