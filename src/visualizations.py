"""
Visualizations Module for E-Waste Policy Pulse.
Generates publication-quality Matplotlib figures using a sustainability-inspired
palette: Forest Green, Emerald, Slate Neutral, Amber, and Crimson.
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sustainability-inspired color palette
COLOR_MAP = {
    "Positive": "#2E7D32",   # Deep forest green
    "Neutral": "#546E7A",    # Slate blue-grey
    "Negative": "#C62828",   # Crimson alert
    "Primary": "#1B5E20",    # Dark green
    "Secondary": "#4CAF50",  # Fresh green
    "Accent": "#FFB300",     # Warm amber
    "Background": "#F9FBF9", # Off-white soft tint
    "Grid": "#E0E0E0",       # Subtle grey
    "Text": "#212121",       # Charcoal text
}

def set_plot_style():
    """Apply consistent styling across matplotlib plots."""
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#CFD8DC"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#ECEFF1"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_sentiment_counts_bar(sentiment_counts: Dict[str, int]) -> plt.Figure:
    """
    Render a clean vertical bar chart of sentiment counts with value labels.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="white")

    categories = ["Positive", "Neutral", "Negative"]
    counts = [sentiment_counts.get(cat, 0) for cat in categories]
    colors = [COLOR_MAP[cat] for cat in categories]

    bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor="none", zorder=3)
    ax.grid(axis="y", zorder=0)

    # Add numeric labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#37474F"
        )

    ax.set_title("Distribution of Posts by Sentiment Category", fontsize=13, fontweight="bold", pad=14, color=COLOR_MAP["Text"])
    ax.set_ylabel("Number of Posts", fontsize=10, color=COLOR_MAP["Text"])
    ax.set_ylim(0, max(counts) * 1.18 if max(counts) > 0 else 10)
    ax.tick_params(colors=COLOR_MAP["Text"], labelsize=10)
    plt.tight_layout()
    return fig


def plot_sentiment_donut(sentiment_percentages: Dict[str, float]) -> plt.Figure:
    """
    Render a modern donut chart of sentiment percentage shares.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor="white")

    categories = ["Positive", "Neutral", "Negative"]
    values = [sentiment_percentages.get(cat, 0.0) for cat in categories]
    colors = [COLOR_MAP[cat] for cat in categories]

    # Donut pie
    wedges, texts, autotexts = ax.pie(
        values,
        labels=categories,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )

    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(10)

    for t in texts:
        t.set_color(COLOR_MAP["Text"])
        t.set_fontsize(10)

    ax.set_title("Sentiment Share Ratio (%)", fontsize=13, fontweight="bold", pad=14, color=COLOR_MAP["Text"])
    plt.tight_layout()
    return fig


def plot_polarity_and_subjectivity(df: pd.DataFrame) -> plt.Figure:
    """
    Plot dual distribution histograms for Polarity and Subjectivity scores.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")

    # Polarity distribution
    ax1.hist(df["polarity"], bins=20, range=(-1.0, 1.0), color=COLOR_MAP["Secondary"], edgecolor="white", alpha=0.85, zorder=3)
    ax1.axvline(0.0, color=COLOR_MAP["Text"], linestyle="--", linewidth=1.2, label="Neutral (0.0)")
    ax1.grid(axis="y", zorder=0)
    ax1.set_title("Polarity Score Spread (-1.0 to +1.0)", fontsize=11, fontweight="bold", color=COLOR_MAP["Text"])
    ax1.set_xlabel("Polarity (Negative ← 0 → Positive)", fontsize=9, color=COLOR_MAP["Text"])
    ax1.set_ylabel("Post Count", fontsize=9, color=COLOR_MAP["Text"])
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    # Subjectivity distribution
    ax2.hist(df["subjectivity"], bins=20, range=(0.0, 1.0), color=COLOR_MAP["Neutral"], edgecolor="white", alpha=0.85, zorder=3)
    ax2.axvline(df["subjectivity"].mean(), color=COLOR_MAP["Accent"], linestyle="--", linewidth=1.2, label=f"Mean ({df['subjectivity'].mean():.2f})")
    ax2.grid(axis="y", zorder=0)
    ax2.set_title("Subjectivity Score Spread (0.0 to 1.0)", fontsize=11, fontweight="bold", color=COLOR_MAP["Text"])
    ax2.set_xlabel("Subjectivity (Objective 0 ← → 1 Subjective)", fontsize=9, color=COLOR_MAP["Text"])
    ax2.set_ylabel("Post Count", fontsize=9, color=COLOR_MAP["Text"])
    ax2.legend(loc="upper right", frameon=False, fontsize=9)

    plt.tight_layout()
    return fig


def plot_top_keywords_bar(top_keywords: List[Tuple[str, int]], title: str = "Top Frequent Terms") -> plt.Figure:
    """
    Horizontal bar chart of top extracted keywords.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")

    if not top_keywords:
        ax.text(0.5, 0.5, "No keyword data available", ha="center", va="center")
        return fig

    # Reverse order so top item is on top of y-axis
    words = [kw[0] for kw in reversed(top_keywords)]
    counts = [kw[1] for kw in reversed(top_keywords)]

    y_pos = np.arange(len(words))
    bars = ax.barh(y_pos, counts, color="#2E7D32", height=0.65, zorder=3)
    ax.grid(axis="x", zorder=0)

    # Labels
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{int(width)}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="#37474F",
            fontweight="bold"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=9.5, color=COLOR_MAP["Text"])
    ax.set_xlabel("Occurrence Frequency in Posts", fontsize=10, color=COLOR_MAP["Text"])
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color=COLOR_MAP["Text"])
    ax.set_xlim(0, max(counts) * 1.15 if max(counts) > 0 else 10)
    plt.tight_layout()
    return fig


def plot_wordcloud_figure(wordcloud_obj) -> plt.Figure:
    """
    Display wordcloud inside a styled Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    ax.imshow(wordcloud_obj, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Corpus Lexical Word Cloud (E-Waste Domain)", fontsize=13, fontweight="bold", pad=12, color=COLOR_MAP["Text"])
    plt.tight_layout()
    return fig


def plot_platform_sentiment_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart showing sentiment counts by social media platform.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")

    if "platform" not in df.columns or "sentiment_label" not in df.columns:
        ax.text(0.5, 0.5, "Platform data not available", ha="center", va="center")
        return fig

    # Group by platform and sentiment
    ct = pd.crosstab(df["platform"], df["sentiment_label"])
    platforms = list(ct.index)

    x = np.arange(len(platforms))
    width = 0.25

    categories = ["Positive", "Neutral", "Negative"]
    for i, cat in enumerate(categories):
        vals = [ct.loc[p, cat] if cat in ct.columns else 0 for p in platforms]
        ax.bar(x + (i - 1) * width, vals, width, label=cat, color=COLOR_MAP[cat], zorder=3)

    ax.grid(axis="y", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontsize=10, color=COLOR_MAP["Text"])
    ax.set_ylabel("Number of Posts", fontsize=10, color=COLOR_MAP["Text"])
    ax.set_title("Sentiment Breakdown Across Social Platforms", fontsize=12, fontweight="bold", pad=12, color=COLOR_MAP["Text"])
    ax.legend(frameon=True, facecolor="white", edgecolor="#CFD8DC", fontsize=9)
    plt.tight_layout()
    return fig


def plot_temporal_sentiment_trend(df: pd.DataFrame) -> plt.Figure:
    """
    Time series chart of monthly post volumes by sentiment.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")

    if "date_parsed" not in df.columns or df["date_parsed"].dropna().empty:
        ax.text(0.5, 0.5, "Temporal date data not available", ha="center", va="center")
        return fig

    valid_dates = df.dropna(subset=["date_parsed"]).copy()
    valid_dates["period"] = valid_dates["date_parsed"].dt.to_period("M").dt.to_timestamp()

    grouped = valid_dates.groupby(["period", "sentiment_label"]).size().unstack(fill_value=0)

    categories = ["Positive", "Neutral", "Negative"]
    for cat in categories:
        if cat in grouped.columns:
            ax.plot(grouped.index, grouped[cat], marker="o", linewidth=2.2, label=cat, color=COLOR_MAP[cat])

    ax.grid(True, zorder=0)
    ax.set_title("Monthly Trend of Posts by Sentiment", fontsize=12, fontweight="bold", pad=12, color=COLOR_MAP["Text"])
    ax.set_xlabel("Timeline (Month)", fontsize=10, color=COLOR_MAP["Text"])
    ax.set_ylabel("Number of Posts", fontsize=10, color=COLOR_MAP["Text"])
    ax.legend(frameon=True, facecolor="white", edgecolor="#CFD8DC", fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_topic_terms_chart(topic_data: dict) -> plt.Figure:
    """
    Horizontal bar chart showing word probabilities for an LDA topic.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor="white")

    words = list(reversed(topic_data.get("words", [])))
    weights = list(reversed(topic_data.get("weights", [])))

    y_pos = np.arange(len(words))
    bars = ax.barh(y_pos, weights, color="#388E3C", height=0.6, zorder=3)
    ax.grid(axis="x", zorder=0)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.3f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#37474F"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=9, color=COLOR_MAP["Text"])
    ax.set_xlabel("Term Probability Weight in Topic", fontsize=9, color=COLOR_MAP["Text"])
    ax.set_title(f"Key Terms: {topic_data.get('topic_label', 'Topic')}", fontsize=11, fontweight="bold", color=COLOR_MAP["Text"])
    ax.set_xlim(0, max(weights) * 1.25 if weights else 0.1)
    plt.tight_layout()
    return fig
