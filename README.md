# ♻️ E-Waste Policy Pulse: Text Mining of E-Waste Policy Social Media Posts Using NLP

[![Python](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLTK-TextBlob-blue.svg)](https://www.nltk.org/)
[![Topic Modeling](https://img.shields.io/badge/Gensim-LDA-orange.svg)](https://radimrehurek.com/gensim/)
[![License](https://img.shields.io/badge/Academic-Educational%20Use-lightgrey.svg)]()

---

## 1. Project Overview & Problem Statement

Electronic waste (e-waste) is one of the fastest-growing environmental and public health challenges globally. In **India**, the world’s third-largest e-waste generator, over 1.6 million tonnes of discarded electronics are produced annually. The Government of India introduced comprehensive regulations—including the **E-Waste (Management) Rules 2016** and the updated **E-Waste (Management) Rules 2022**—mandating **Extended Producer Responsibility (EPR)** for brand electronics manufacturers.

Despite these regulatory frameworks, over **90% of India's e-waste continues to be processed within the unorganized/informal sector** (e.g. Seelampur in Delhi, Moradabad in Uttar Pradesh). Informal dismantling often relies on hazardous methods such as open-air circuit board burning and crude acid leaching to extract precious metals, releasing toxic heavy metals (lead, mercury, cadmium) into local soil, groundwater, and air.

**E-Waste Policy Pulse** is an academic Natural Language Processing (NLP) web application that mines public social media discussions (Twitter/X, LinkedIn, Reddit, Instagram, Facebook) to extract actionable policy insights. The interactive dashboard allows environmental researchers, policy analysts, and citizens to monitor public perception, evaluate awareness of EPR rules, and identify recurring community grievances.

---

## 2. Research Objectives

1. **Quantify Public Sentiment:** Determine the proportion of online discussions reflecting positive reception (e.g., successful municipal collection drives, circular economy initiatives), neutral announcements (CPCB circulars, gazette notifications), and negative sentiment (complaints of toxic smoke, non-functioning collection points, corporate greenwashing).
2. **Identify Salient Policy Lexicon:** Discover the most frequent terms and phrases characterizing e-waste policy in India.
3. **Analyze Recurring Thematic Clusters:** Employ unsupervised topic modelling (Latent Dirichlet Allocation) to uncover underlying discussions without human bias.
4. **Evaluate Cross-Platform Dynamics:** Compare discourse tones between platforms (e.g. professional policy focus on LinkedIn vs. citizen grievances on Reddit and Twitter).
5. **Provide Empirical Evidence for Policy Refinement:** Surface representative citizen posts illustrating ground-level barriers to safe e-waste disposal.

---

## 3. Technology Stack (Strictly Implemented)

| Technology | Purpose | Role in Project |
|---|---|---|
| **Python 3.11** | Core Language | Core processing and orchestration |
| **Pandas** | Data Ingestion & Analysis | Schema validation, deduplication, filtering, aggregation |
| **NLTK** | NLP Preprocessing | Tokenisation, stopword removal, lemmatisation, negation preservation |
| **TextBlob** | Baseline Sentiment | Polarity ([-1.0, 1.0]) and subjectivity ([0.0, 1.0]) calculation |
| **Matplotlib** | Visualisation | Sustainability-themed publication charts (bars, donuts, histograms) |
| **WordCloud** | Lexical Visualisation | High-resolution visual representation of corpus vocabulary |
| **Streamlit** | Web Interface | Interactive 5-page research dashboard |
| **Gensim** | Topic Modelling (Extension) | Latent Dirichlet Allocation (LDA) for computational theme discovery |
| **Pytest** | Testing Suite | Unit tests for preprocessing, sentiment classification, loader, and LDA |

> **Note on Architecture:** No external heavy frameworks (React, FastAPI, Spring Boot, external databases, or paid APIs) were introduced, keeping the project 100% lightweight, reproducible, and easy to run locally.

---

## 4. Project Architecture & File Structure

```
e_waste_project/
├── app.py                     # Main Streamlit web application & multi-page navigation
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation and viva preparation guide
├── .gitignore                 # Git ignore file for Python, Streamlit, and OS files
├── data/
│   └── sample_posts.csv       # Demonstration dataset (synthetic, India e-waste focus)
├── src/
│   ├── __init__.py            # Package initializer
│   ├── data_loader.py         # CSV loading, column standardisation, deduplication & validation
│   ├── preprocessing.py       # Cleaning, tokenisation, negation preservation & lemmatisation
│   ├── sentiment_analysis.py  # TextBlob scoring, threshold rules, and metric aggregation
│   ├── keyword_analysis.py    # Unigram/bigram frequency, domain lexicon & word cloud builder
│   ├── topic_modeling.py      # Gensim LDA topic extraction with Scipy compatibility patch
│   └── visualizations.py      # Sustainability-themed Matplotlib figures (green, slate, crimson)
└── tests/
    ├── __init__.py            # Test package initializer
    ├── test_data_loader.py    # Unit tests for CSV ingestion, schema validation & deduplication
    ├── test_preprocessing.py  # Unit tests for text cleaning, negations & lemmatisation
    ├── test_sentiment.py      # Unit tests for polarity rules & metric consistency
    └── test_topic_modeling.py # Unit tests for Gensim LDA & small-corpus guards
```

---

## 5. Dataset Format & Provenance

### Supported Schema

The application automatically maps and validates the following columns:

| Column | Type | Mandatory? | Description |
|---|---|---|---|
| `post_id` | String | Auto-generated if absent | Unique identifier for each post (e.g. `P1001`) |
| `text` | String | **Yes** | Social media post content |
| `date` | Date/String | Optional | Publication timestamp (e.g. `2024-03-15`) |
| `platform` | String | Optional | Source network (Twitter, LinkedIn, Reddit, Facebook, Instagram) |
| `likes` | Integer | Optional | Engagement metrics / like count |

### Demonstration Dataset Provenance

- The included `data/sample_posts.csv` contains realistic social media posts specifically crafted for demonstration.
- **Academic Transparency Notice:** The default sample posts are synthetic demonstration records curated to model real public discourse observed in Central Pollution Control Board (CPCB) audits, Toxics Link studies, and MoEFCC gazette notifications. They are explicitly labelled in the interface as demonstration data.
- The raw sample file includes deliberate test cases (a duplicate entry and an empty text entry) to demonstrate automated cleaning and validation statistics in the UI.

---

## 6. Installation & Execution Guide

### Step 1: Clone or Navigate to the Project Directory

```bash
cd "/Users/tanishad/Desktop/e waste project"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Automated Unit Test Suite

```bash
pytest tests/ -v
```
*(All 19 tests will execute, validating text cleaning, negation handling, sentiment thresholds, and topic modeling).*

### Step 4: Launch the Streamlit Application

```bash
streamlit run app.py
```

Open your web browser and navigate to:  
👉 **`http://localhost:8501`**

---

## 7. Explanation of the NLP Pipeline

```
[Raw Post Text]
      │
      ▼
1. Structural Cleaning (Regex & HTML Unescaping)
   - Strips URLs (http/https/www)
   - Removes HTML entities (<...>, &amp;)
   - Removes @mentions, preserves hashtag text
   - Normalises e-waste variants ("e waste", "ewaste" -> "e-waste")
      │
      ▼
2. Case Normalisation & Tokenisation
   - Converts text to lowercase
   - Tokenises into individual word tokens
      │
      ▼
3. Negation-Aware Stop-Word Filtering
   - Standard English stopwords removed (e.g., 'the', 'is', 'at')
   - CRITICAL: Preserves negation terms ('not', 'no', 'never', 'without', 'cannot', 'barely')
     Preventing artificial sentiment reversal (e.g. "not working" != "working")
      │
      ▼
4. Morphological Lemmatisation
   - Applies NLTK WordNetLemmatizer
   - Maps plurals and verb inflections to root forms ("batteries" -> "battery", "recyclers" -> "recycler")
      │
      ▼
5. Downstream NLP Analysis
   ├── TextBlob Sentiment Evaluation (Polarity & Subjectivity)
   ├── N-gram Frequency & Domain Terminology Mapping
   └── Gensim LDA Probabilistic Topic Modeling
```

---

## 8. Sentiment Analysis Methodology

### Mathematical Definition & Thresholds

Sentiment scoring uses TextBlob's pattern-matching lexicon:
- **Polarity ($P$):** Float in the range $[-1.0, +1.0]$, measuring the degree of positive or negative emotion.
- **Subjectivity ($S$):** Float in the range $[0.0, 1.0]$, measuring factual objective statements ($0.0$) versus opinionated personal beliefs ($1.0$).

### Classification Rule:

$$\text{Sentiment Label} = \begin{cases} 
\text{Positive}, & \text{if } P > 0.0 \\ 
\text{Negative}, & \text{if } P < 0.0 \\ 
\text{Neutral},  & \text{if } P = 0.0 
\end{cases}$$

### Academic Disclaimer & Methodological Limitations:

1. **Lexicon-Based Baseline:** TextBlob relies on a fixed English lexicon. It is **not** a transformer model (like RoBERTa) fine-tuned on environmental jurisprudence.
2. **Environmental Domain Bias:** Words such as *hazard*, *acid*, *waste*, or *toxic* carry default negative polarity in general lexicons, even when used in objective engineering or regulatory statements (e.g. *"Standard operating procedure for safe acid neutralisation"*).
3. **Sarcasm & Context:** Sarcasm and regional Indian idioms (e.g. *"Great job by brand dumping scrap in open drain"*) can cause false positive classifications.

---

## 9. Topic Modelling Methodology (Gensim LDA)

### Probabilistic Formulation

Latent Dirichlet Allocation (LDA) is a generative probabilistic model of a corpus. Documents are represented as random mixtures over latent topics, and each topic is characterized by a Dirichlet distribution over words:

$$p(w \mid d) = \sum_{t=1}^{K} p(w \mid t) \cdot p(t \mid d)$$

Where:
- $K$: User-configured number of topics (slider from 2 to 6).
- $p(w \mid t)$: Probability of word $w$ occurring in topic $t$.
- $p(t \mid d)$: Proportion of topic $t$ in document $d$.

### Human-in-the-Loop Interpretation

LDA groups words based purely on statistical co-occurrence. In this project:
- **Topic 1 (Infrastructure & Kiosks):** Words: *collection, drive, centre, municipal, drop-off, battery, smartphone*.
- **Topic 2 (Regulatory & Compliance):** Words: *epr, rules, producer, cpcb, portal, targets, responsibility*.
- **Topic 3 (Environmental Health & Informal Sector):** Words: *toxic, informal, smoke, seelampur, lead, acid, moradabad*.

---

## 10. Walkthrough of Dashboard Pages

### 🌐 Page 1: Overview & Executive Summary
- **KPI Metrics:** Total posts analyzed, counts and percentages for Positive, Neutral, and Negative sentiments, average polarity.
- **Data-Driven Summary:** Dynamically written summary derived from actual calculated metrics.
- **Visualisations:** Side-by-side count bar chart and percentage donut chart.
- **Key Terms Preview:** Top 10 high-frequency policy words.

### 📊 Page 2: Sentiment Analysis
- **Score Distributions:** Dual histograms showing polarity spread ([-1.0 to +1.0]) and subjectivity distribution.
- **Cross-Platform Breakdown:** Grouped bar chart comparing sentiment ratios across Twitter, LinkedIn, Reddit, Facebook, and Instagram.
- **Temporal Trends:** Time-series tracking of monthly post volumes by sentiment category.
- **Interactive Post Explorer:** Searchable table with filtering by sentiment label, platform, and text queries.

### 🔍 Page 3: Keyword & Text Mining
- **Word Cloud:** Sustainability green lexical cloud highlighting the most prominent terms.
- **Unigram & Bigram Analysis:** Top 15 single words and top 15 2-word phrases.
- **Domain Lexicon Table:** Pre-configured matrix of 24 e-waste policy terms showing corpus occurrences, dataset percentage, and positive/negative distribution.
- **Interactive Keyword Inspector:** Search any term (e.g. *EPR*, *kabadiwala*, *battery*) to view matching posts and sentiment share.

### 🧩 Page 4: Topic Analysis (Gensim LDA)
- **Hyperparameter Slider:** Allows users to adjust topic count ($k$) between 2 and 6.
- **Topic Cards & Word Weights:** Visualizes top terms and their probability weights per topic.
- **Dominant Topic Share:** Table showing how documents distribute across topics.
- **Representative Documents:** Displays the highest-confidence posts for each discovered topic.

### 📁 Page 5: Dataset Explorer & Research Insights
- **Full Enriched Table:** Direct comparison between raw text, cleaned tokens, and sentiment scores.
- **CSV Export:** Downloadable CSV including all generated NLP features.
- **Research Insights Section:** Answers the 5 core research questions with live data calculations.
- **Limitations & Ethical Scope:** Notes sampling bias, platform representation, and social media constraints in India.

---

## 11. Code Walkthrough for Academic Viva Voce

| Module | What to explain to the examiner |
|---|---|
| `src/data_loader.py` | Explain how the loader standardises varying column headers (using aliases for `likes`, `created_at`, `content`), handles empty files gracefully, and deduplicates identical texts while logging auditing statistics. |
| `src/preprocessing.py` | Emphasize why negations (`not`, `no`, `never`) were **preserved** instead of stripped with standard stopwords. Explain the role of `WordNetLemmatizer` in reducing inflected words to base dictionary forms. |
| `src/sentiment_analysis.py` | Explain TextBlob's polarity formula, justify the explicit thresholds ($>0$, $<0$, $==0$), and present the academic limitations regarding sarcasm and environmental technical terms. |
| `src/keyword_analysis.py` | Explain n-gram token aggregation, domain dictionary tagging, and why raw word frequency indicates topical salience rather than public approval. |
| `src/topic_modeling.py` | Explain bag-of-words vectorization, dictionary filtering, the Latent Dirichlet Allocation algorithm, and the Scipy 1.15 compatibility patch (`scipy.linalg.triu = np.triu`). |
| `src/visualizations.py` | Point out the sustainability-inspired color scheme (Forest Green `#2E7D32`, Slate `#546E7A`, Crimson `#C62828`) and how clean Matplotlib figures integrate into Streamlit. |
| `tests/` | Demonstrate that the project has 19 unit tests passing across all critical components, proving robustness and reliability. |

---

## 12. Limitations & Future Scope

1. **Transformer Fine-Tuning:** Future work can replace TextBlob with a domain-adapted transformer (e.g. `climate-bert` or fine-tuned `RoBERTa-large`) trained on Indian environmental law and social posts.
2. **Multilingual NLP:** India has 22 official languages. Incorporating Indic NLP models (e.g., IndicBERT) to mine Hindi, Tamil, Telugu, and Bengali posts will significantly expand coverage.
3. **Formalizing Informal Sector Voice:** Social media predominantly captures urban English-speaking users. Field surveys and voice-to-text NLP are recommended to capture voices from informal scrap workers directly.

---

## 13. License & Academic Attribution

Developed for academic research and educational evaluation on Natural Language Processing for Environmental Policy Analysis.
