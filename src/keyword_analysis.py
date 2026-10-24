"""
Keyword and Text Mining Module for E-Waste Policy Pulse.
Extracts top meaningful terms, computes domain-specific terminology frequencies,
generates word clouds, and supports keyword-based post inspection.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# Configurable domain-specific lexicon covering Indian e-waste policy, recycling, and hazards
DEFAULT_DOMAIN_TERMS = [
    "e-waste",
    "recycling",
    "epr",
    "cpcb",
    "informal",
    "kabadiwala",
    "collection",
    "pollution",
    "toxic",
    "lead",
    "mercury",
    "battery",
    "batteries",
    "producer",
    "responsibility",
    "disposal",
    "refurbishment",
    "take-back",
    "greenwashing",
    "circular",
    "compliance",
    "landfill",
    "scrap",
    "electronics"
]

KEYWORD_ACADEMIC_DISCLAIMER = """
Academic Methodological Note:
Word frequency indicates topical salience (what subjects appear frequently in the discourse),
but does not, on its own, prove public consensus, policy effectiveness, or emotional endorsement.
High frequency of a term like 'recycling' can occur in both positive announcements of new kiosks
and sharp criticisms of non-functioning recycling centres. Text mining insights must always be
interpreted in conjunction with sentiment analysis, topic context, and qualitative post inspection.
"""


def get_top_unigrams(
    df: pd.DataFrame,
    top_n: int = 20,
    min_length: int = 3,
    custom_stopwords: Optional[set] = None
) -> List[Tuple[str, int]]:
    """
    Extract the most frequent meaningful unigrams from tokenised posts.
    """
    if "tokens" not in df.columns or df.empty:
        return []

    extra_stops = custom_stopwords or set()
    counter = Counter()

    for tokens in df["tokens"]:
        if isinstance(tokens, list):
            for tok in tokens:
                t = tok.lower().strip()
                if len(t) >= min_length and t not in extra_stops:
                    counter[t] += 1

    return counter.most_common(top_n)


def get_top_bigrams(
    df: pd.DataFrame,
    top_n: int = 15,
    custom_stopwords: Optional[set] = None
) -> List[Tuple[str, int]]:
    """
    Extract the most frequent meaningful 2-word phrases (bigrams) from tokens.
    """
    if "tokens" not in df.columns or df.empty:
        return []

    extra_stops = custom_stopwords or set()
    counter = Counter()

    for tokens in df["tokens"]:
        if isinstance(tokens, list) and len(tokens) >= 2:
            filtered = [t.lower().strip() for t in tokens if len(t) >= 2 and t not in extra_stops]
            for i in range(len(filtered) - 1):
                bg = f"{filtered[i]} {filtered[i+1]}"
                counter[bg] += 1

    return counter.most_common(top_n)


def search_posts_by_keyword(df: pd.DataFrame, keyword: str) -> Dict[str, Any]:
    """
    Search for posts containing a specific keyword or phrase.

    Returns:
    --------
    dict containing:
        - 'keyword': str
        - 'match_count': int
        - 'total_posts': int
        - 'match_percentage': float
        - 'matching_df': pd.DataFrame
        - 'sentiment_breakdown': dict
    """
    if not keyword or not keyword.strip() or df.empty:
        return {
            "keyword": keyword,
            "match_count": 0,
            "total_posts": len(df),
            "match_percentage": 0.0,
            "matching_df": pd.DataFrame(),
            "sentiment_breakdown": {},
        }

    k = keyword.strip().lower()
    total = len(df)

    # Search in both raw text and cleaned text
    match_mask = (
        df["text"].str.lower().str.contains(k, regex=False, na=False) |
        (df["cleaned_text"].str.lower().str.contains(k, regex=False, na=False) if "cleaned_text" in df.columns else False)
    )

    matching_df = df[match_mask].copy().reset_index(drop=True)
    match_count = len(matching_df)
    pct = round((match_count / total) * 100, 2) if total > 0 else 0.0

    sentiment_breakdown = {}
    if "sentiment_label" in matching_df.columns:
        sentiment_breakdown = matching_df["sentiment_label"].value_counts().to_dict()

    return {
        "keyword": keyword,
        "match_count": match_count,
        "total_posts": total,
        "match_percentage": pct,
        "matching_df": matching_df,
        "sentiment_breakdown": sentiment_breakdown,
    }


def analyze_domain_keywords(df: pd.DataFrame, domain_terms: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Assess frequency of key e-waste policy domain terms across the entire dataset.
    """
    terms = domain_terms if domain_terms is not None else DEFAULT_DOMAIN_TERMS
    records = []
    total = len(df)

    for term in terms:
        search_res = search_posts_by_keyword(df, term)
        pos = search_res["sentiment_breakdown"].get("Positive", 0)
        neg = search_res["sentiment_breakdown"].get("Negative", 0)
        neu = search_res["sentiment_breakdown"].get("Neutral", 0)

        records.append({
            "Domain Keyword": term,
            "Post Occurrences": search_res["match_count"],
            "Dataset Share (%)": search_res["match_percentage"],
            "Positive Posts": pos,
            "Negative Posts": neg,
            "Neutral Posts": neu
        })

    result_df = pd.DataFrame(records).sort_values(by="Post Occurrences", ascending=False).reset_index(drop=True)
    return result_df


def build_wordcloud_object(
    corpus_text: str,
    width: int = 800,
    height: int = 400,
    background_color: str = "#FAFAF9",
    colormap: str = "Greens",
    max_words: int = 100
) -> WordCloud:
    """
    Build a WordCloud instance from corpus text.
    Uses sustainability green theme by default.
    """
    if not corpus_text or not corpus_text.strip():
        corpus_text = "e-waste recycling sustainability environment policy management"

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        collocations=False,
        random_state=42
    )
    wc.generate(corpus_text)
    return wc
