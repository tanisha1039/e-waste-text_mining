"""
NLP Preprocessing Module for E-Waste Policy Pulse.
Performs cleaning, tokenisation, negation-aware stopword removal, and lemmatisation.
Maintains original text alongside processed tokens and cleaned representations.
"""

import html
import re
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set of critical negations and polarity modifiers that MUST be preserved
# to avoid reversing or distorting text sentiment.
PRESERVED_NEGATIONS: Set[str] = {
    "not", "no", "nor", "neither", "never", "none", "barely",
    "hardly", "scarcely", "cannot", "without", "against", "lack", "lacking"
}

# Domain-specific terms that should be preserved as unified concepts
DOMAIN_KEYWORDS: Set[str] = {
    "e-waste", "ewaste", "epr", "cpcb", "spcb", "rohs", "pcb", "crt",
    "kabadiwala", "recycler", "recycling", "landfill", "leachate"
}

_NLTK_INITIALIZED = False
_STOPWORDS_SET: Set[str] = set()
_LEMMATIZER = None


def ensure_nltk_resources() -> bool:
    """
    Ensure essential NLTK resources are available locally.
    Uses offline resources when available, with instant fallback to avoid network hangs.
    """
    global _NLTK_INITIALIZED, _STOPWORDS_SET, _LEMMATIZER
    if _NLTK_INITIALIZED:
        return True

    # Initialize Stopwords
    try:
        raw_stops = set(stopwords.words("english"))
        _STOPWORDS_SET = raw_stops - PRESERVED_NEGATIONS
    except Exception:
        # Fallback minimal stop words if NLTK stopwords unavailable
        fallback_stops = {
            "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
            "he", "him", "his", "she", "her", "it", "its", "they", "them",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "into", "through",
            "during", "before", "after", "above", "below", "to", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "only", "own", "same", "so", "than", "too", "very",
            "s", "t", "can", "will", "just", "don", "should", "now"
        }
        _STOPWORDS_SET = fallback_stops - PRESERVED_NEGATIONS

    # Initialize Lemmatizer
    try:
        lemmatizer = WordNetLemmatizer()
        # Verify it works
        lemmatizer.lemmatize("batteries")
        _LEMMATIZER = lemmatizer
    except Exception:
        _LEMMATIZER = None

    _NLTK_INITIALIZED = True
    return True


def clean_text_structure(raw_text: str) -> str:
    """
    Remove URLs, HTML entities, user handles, excess whitespace and unwanted symbols.
    Preserves alphanumeric terms and hyphenated domain terms like e-waste.
    """
    if not isinstance(raw_text, str):
        return ""

    # Decode HTML entities (e.g., &amp; -> &)
    text = html.unescape(raw_text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs (http/https/www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove user mentions (e.g. @user)
    text = re.sub(r"@\w+", " ", text)

    # Clean hashtag symbol but keep text (e.g. #RecycleIndia -> RecycleIndia)
    text = re.sub(r"#(\w+)", r"\1", text)

    # Replace emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Normalize e-waste variants ('e-waste', 'e waste', 'ewaste') to canonical 'e-waste'
    text = re.sub(r"\be\s*[-]?\s*waste\b", "e-waste", text, flags=re.IGNORECASE)

    # Remove numbers if pure digits or symbols, but allow words
    text = re.sub(r"[^\w\s-]", " ", text)

    # Collapse multiple whitespaces and hyphens
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenise text into words using NLTK word_tokenize with regex fallback.
    """
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback regex tokenisation
        tokens = re.findall(r"\b[\w-]+\b", text)
    return tokens


def preprocess_post(
    text: str,
    lemmatize: bool = True,
    remove_stopwords: bool = True
) -> dict:
    """
    Preprocess a single social media post text.

    Returns:
    --------
    dict containing:
        - 'cleaned_text': str
        - 'tokens': List[str]
    """
    ensure_nltk_resources()

    # 1. Structure cleaning
    cleaned = clean_text_structure(text)

    # 2. Lowercasing
    cleaned_lower = cleaned.lower()

    # 3. Tokenisation
    raw_tokens = tokenize_text(cleaned_lower)

    # 4. Filter tokens: alphanumeric, drop single letter non-words except 'a'/'i' (or drop all <=1 len)
    valid_tokens = []
    for token in raw_tokens:
        token_clean = token.strip("-_")
        if not token_clean or len(token_clean) < 2:
            continue
        # Check if purely digits
        if token_clean.isdigit():
            continue
        valid_tokens.append(token_clean)

    # 5. Stopword removal (preserving negations)
    if remove_stopwords:
        filtered_tokens = [tok for tok in valid_tokens if tok not in _STOPWORDS_SET]
    else:
        filtered_tokens = valid_tokens

    # 6. Lemmatisation
    if lemmatize and _LEMMATIZER is not None:
        final_tokens = []
        for tok in filtered_tokens:
            if tok in PRESERVED_NEGATIONS or tok in DOMAIN_KEYWORDS:
                final_tokens.append(tok)
            else:
                try:
                    # Lemmatize noun then verb
                    lemmatized = _LEMMATIZER.lemmatize(tok, pos="n")
                    lemmatized = _LEMMATIZER.lemmatize(lemmatized, pos="v")
                    final_tokens.append(lemmatized)
                except Exception:
                    final_tokens.append(tok)
    else:
        final_tokens = filtered_tokens

    cleaned_str = " ".join(final_tokens)

    return {
        "cleaned_text": cleaned_str,
        "tokens": final_tokens
    }


def preprocess_dataframe(
    df,
    text_column: str = "text",
    lemmatize: bool = True,
    remove_stopwords: bool = True
):
    """
    Preprocess entire DataFrame while keeping the original text column unchanged.
    Adds 'cleaned_text', 'tokens', and 'token_count' columns.
    """
    ensure_nltk_resources()

    cleaned_texts = []
    tokens_list = []
    token_counts = []

    for raw_text in df[text_column]:
        res = preprocess_post(raw_text, lemmatize=lemmatize, remove_stopwords=remove_stopwords)
        cleaned_texts.append(res["cleaned_text"])
        tokens_list.append(res["tokens"])
        token_counts.append(len(res["tokens"]))

    df["cleaned_text"] = cleaned_texts
    df["tokens"] = tokens_list
    df["token_count"] = token_counts

    return df
