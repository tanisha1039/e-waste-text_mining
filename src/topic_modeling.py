"""
Topic Modelling Module for E-Waste Policy Pulse.
Implements Latent Dirichlet Allocation (LDA) using Gensim.
Safely extracts computational topic-word distributions and representative posts.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Compatibility patch for scipy >= 1.13 and gensim < 4.3.3
try:
    import scipy.linalg
    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = np.triu
except Exception:
    pass

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


LDA_ACADEMIC_DISCLAIMER = """
Academic Topic Modelling Methodology & Interpretation:
Latent Dirichlet Allocation (LDA) is an unsupervised generative probabilistic model.
It discovers latent themes by assuming each post is a mixture of topics, and each topic is a
probability distribution over words in the vocabulary.
Key Research Principles:
1. Computational Discovery: The algorithm groups co-occurring terms purely based on statistical patterns.
   It does not understand semantics or policy implications on its own.
2. Human-in-the-Loop Interpretation: The assigned descriptive theme labels are human interpretations
   derived by inspecting the top representative words and dominant posts.
3. Dataset Sensitivity: Small corpora (<50 documents) or very short texts can lead to overlapping
   or noisy topic distributions. Topic coherence should be evaluated with caution.
"""


def perform_lda_topic_modeling(
    df: pd.DataFrame,
    num_topics: int = 3,
    top_words_count: int = 8,
    min_tokens: int = 3,
    passes: int = 15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fit an LDA model on the tokenised dataset.

    Returns:
    --------
    dict containing:
        - success: bool
        - message: str
        - num_topics: int
        - topics: List[dict] (topic_id, top_words, top_weights, formatted_string)
        - dominant_df: pd.DataFrame with 'dominant_topic', 'topic_probability'
        - topic_distribution: dict
        - representative_posts: dict of topic_id -> list of top posts
    """
    if not GENSIM_AVAILABLE:
        return {
            "success": False,
            "message": "Gensim library is not available in the current Python environment.",
            "topics": [],
        }

    if df.empty or "tokens" not in df.columns:
        return {
            "success": False,
            "message": "Dataset is empty or lacks tokenised text.",
            "topics": [],
        }

    # Filter out posts with very few tokens
    valid_mask = df["tokens"].apply(lambda toks: isinstance(toks, list) and len(toks) >= min_tokens)
    modeling_df = df[valid_mask].copy().reset_index(drop=True)

    if len(modeling_df) < 10:
        return {
            "success": False,
            "message": f"Dataset contains only {len(modeling_df)} documents meeting minimum token requirements ({min_tokens}+ words). LDA requires at least 10 documents for meaningful convergence.",
            "topics": [],
        }

    # Safe bound on number of topics
    actual_num_topics = max(2, min(num_topics, len(modeling_df)))

    tokens_list = modeling_df["tokens"].tolist()

    # Create Dictionary and Corpus
    dictionary = corpora.Dictionary(tokens_list)

    # Filter extreme tokens if vocabulary is sufficiently large
    if len(dictionary) > 30:
        dictionary.filter_extremes(no_below=2, no_above=0.85)

    if len(dictionary) == 0:
        return {
            "success": False,
            "message": "Vocabulary is too sparse after token filtering.",
            "topics": [],
        }

    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

    # Find non-empty bow documents to train LDA
    training_corpus = [bow for bow in corpus if len(bow) > 0]
    if len(training_corpus) < 5:
        return {
            "success": False,
            "message": "Too few non-empty bag-of-words vectors to build LDA model.",
            "topics": [],
        }

    # Fit LDA
    try:
        lda_model = LdaModel(
            corpus=training_corpus,
            id2word=dictionary,
            num_topics=actual_num_topics,
            random_state=random_state,
            passes=passes,
            alpha="auto",
            eta="auto",
            per_word_topics=False
        )
    except Exception as exc:
        return {
            "success": False,
            "message": f"LDA model fitting error: {str(exc)}",
            "topics": [],
        }

    # Extract topic terms
    topics_data = []
    for topic_id in range(actual_num_topics):
        terms_weights = lda_model.show_topic(topic_id, topn=top_words_count)
        words = [term for term, _ in terms_weights]
        weights = [round(float(weight), 4) for _, weight in terms_weights]
        summary_str = ", ".join([f"{w} ({wt:.3f})" for w, wt in zip(words, weights)])

        topics_data.append({
            "topic_id": topic_id,
            "topic_label": f"Topic #{topic_id + 1}",
            "words": words,
            "weights": weights,
            "terms_weights": terms_weights,
            "summary_str": summary_str
        })

    # Assign dominant topic to every document in modeling_df
    dominant_topics = []
    topic_probabilities = []

    for bow in corpus:
        if len(bow) > 0:
            topic_dist = lda_model.get_document_topics(bow)
            if topic_dist:
                sorted_dist = sorted(topic_dist, key=lambda x: x[1], reverse=True)
                dominant_topics.append(sorted_dist[0][0])
                topic_probabilities.append(round(float(sorted_dist[0][1]), 4))
            else:
                dominant_topics.append(-1)
                topic_probabilities.append(0.0)
        else:
            dominant_topics.append(-1)
            topic_probabilities.append(0.0)

    modeling_df["dominant_topic"] = dominant_topics
    modeling_df["topic_probability"] = topic_probabilities
    modeling_df["dominant_topic_label"] = modeling_df["dominant_topic"].apply(
        lambda tid: f"Topic #{tid + 1}" if tid >= 0 else "Unassigned"
    )

    # Calculate topic distribution
    counts = modeling_df["dominant_topic_label"].value_counts().to_dict()

    # Extract top representative posts per topic (highest probability)
    representative_posts = {}
    for topic_id in range(actual_num_topics):
        sub_df = modeling_df[modeling_df["dominant_topic"] == topic_id]
        if not sub_df.empty:
            top_reps = sub_df.sort_values(by="topic_probability", ascending=False).head(3)
            available_cols = [c for c in ["post_id", "text", "topic_probability", "sentiment_label"] if c in top_reps.columns]
            representative_posts[topic_id] = top_reps[available_cols].to_dict(orient="records")
        else:
            representative_posts[topic_id] = []

    return {
        "success": True,
        "message": f"Successfully extracted {actual_num_topics} topics using Gensim LDA.",
        "num_topics": actual_num_topics,
        "topics": topics_data,
        "dominant_df": modeling_df,
        "topic_distribution": counts,
        "representative_posts": representative_posts,
    }
