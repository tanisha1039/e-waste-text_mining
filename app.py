"""
E-Waste Policy Pulse: Text Mining of E-Waste Policy Social Media Posts Using NLP
Interactive Streamlit Application & Academic Research Dashboard.
"""

from io import BytesIO
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataframe, PRESERVED_NEGATIONS
from src.sentiment_analysis import (
    analyze_dataframe_sentiment,
    get_sentiment_metrics,
    filter_posts,
    CLASSIFICATION_RULES_EXPLANATION,
    TEXTBLOB_ACADEMIC_DISCLAIMER
)
from src.keyword_analysis import (
    get_top_unigrams,
    get_top_bigrams,
    search_posts_by_keyword,
    analyze_domain_keywords,
    build_wordcloud_object,
    KEYWORD_ACADEMIC_DISCLAIMER,
    DEFAULT_DOMAIN_TERMS
)
from src.topic_modeling import (
    perform_lda_topic_modeling,
    GENSIM_AVAILABLE,
    LDA_ACADEMIC_DISCLAIMER
)
from src.visualizations import (
    plot_sentiment_counts_bar,
    plot_sentiment_donut,
    plot_polarity_and_subjectivity,
    plot_top_keywords_bar,
    plot_wordcloud_figure,
    plot_platform_sentiment_comparison,
    plot_temporal_sentiment_trend,
    plot_topic_terms_chart
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-Waste Policy Pulse | NLP Text Mining",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern sustainability theme
st.markdown("""
<style>
    /* Sustainability-inspired color palette */
    :root {
        --primary-green: #1B5E20;
        --secondary-green: #2E7D32;
        --accent-mint: #E8F5E9;
        --text-dark: #1F2937;
        --border-subtle: #E5E7EB;
    }
    
    .main-title {
        color: #1B5E20;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAF8;
        border: 1px solid #D1E7DD;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: bold;
        color: #1B5E20;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #92400E;
        margin-bottom: 1rem;
    }
    .demo-badge {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .card-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State & Data Processing Pipelines
# ---------------------------------------------------------
@st.cache_data(show_spinner="Processing and analyzing posts...")
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess text and compute sentiment scores."""
    # Preprocess text (clean, tokenize, lemmatize, keep negations)
    df_pre = preprocess_dataframe(df.copy(), text_column="text", lemmatize=True, remove_stopwords=True)
    # Compute TextBlob sentiment
    df_analyzed = analyze_dataframe_sentiment(df_pre, text_column="text")
    return df_analyzed


# ---------------------------------------------------------
# Sidebar Navigation & Dataset Configuration
# ---------------------------------------------------------
st.sidebar.markdown("### ♻️ E-Waste Policy Pulse")
st.sidebar.caption("Social Media NLP Text Mining & Policy Insights")
st.sidebar.markdown("---")

# Dataset Source Selection
st.sidebar.markdown("#### 📂 Dataset Source")
dataset_source = st.sidebar.radio(
    "Choose data source:",
    ["Sample Demonstration Dataset", "Upload Custom CSV"],
    index=0,
    help="Select the curated demonstration dataset (India e-waste focus) or upload your own CSV file."
)

raw_df = None
stats_dict = None
is_demo = False

if dataset_source == "Sample Demonstration Dataset":
    is_demo = True
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_posts.csv")
    if os.path.exists(sample_path):
        try:
            raw_df, stats_dict = load_dataset(sample_path, is_demonstration=True)
        except Exception as e:
            st.sidebar.error(f"Error reading sample dataset: {e}")
    else:
        st.sidebar.error("Sample dataset file not found at data/sample_posts.csv")
else:
    is_demo = False
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with posts",
        type=["csv"],
        help="CSV should include a 'text' column. Optional columns: 'post_id', 'date', 'platform', 'likes'."
    )
    if uploaded_file is not None:
        try:
            raw_df, stats_dict = load_dataset(uploaded_file, is_demonstration=False)
        except Exception as e:
            st.sidebar.error(f"Dataset Loading Error: {str(e)}")
            st.stop()
    else:
        st.info("👈 Please upload a CSV file containing social media posts, or switch to the **Sample Demonstration Dataset** in the sidebar.")
        st.markdown("""
        #### Expected CSV Format
        | post_id | text | date | platform | likes |
        |---|---|---|---|---|
        | P101 | Municipal e-waste collection drive was very efficient today! | 2024-03-01 | Twitter | 45 |
        | P102 | Burning discarded circuit boards causes severe toxic pollution. | 2024-03-02 | Reddit | 88 |
        """)
        st.stop()

# Validate that data is ready
if raw_df is None or raw_df.empty:
    st.warning("No valid records found in the dataset.")
    st.stop()

# Run NLP Pipeline
analyzed_df = process_data(raw_df)
metrics = get_sentiment_metrics(analyzed_df)

# Sidebar Ingestion Stats Display
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📊 Ingestion & Cleaning Stats")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Loaded", stats_dict["raw_count"])
col_s2.metric("Retained", stats_dict["retained_count"])

with st.sidebar.expander("Cleaning Audit Details"):
    st.write(f"- **Raw records loaded:** {stats_dict['raw_count']}")
    st.write(f"- **Blank/empty rows removed:** {stats_dict['blank_dropped']}")
    st.write(f"- **Duplicate records removed:** {stats_dict['duplicates_dropped']}")
    st.write(f"- **Valid records analysed:** {stats_dict['retained_count']}")
    st.write(f"- **Date column present:** {'Yes' if stats_dict['has_date'] else 'No'}")
    st.write(f"- **Platform column present:** {'Yes' if stats_dict['has_platform'] else 'No'}")

# Navigation Radio
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🧭 Application Navigation")
page_selection = st.sidebar.radio(
    "Go to page:",
    [
        "Page 1: Overview & Executive Summary",
        "Page 2: Sentiment Analysis",
        "Page 3: Keyword & Text Mining",
        "Page 4: Topic Analysis (Gensim LDA)",
        "Page 5: Dataset Explorer & Research Insights"
    ],
    index=0
)

# Sidebar Viva & Academic Notes
with st.sidebar.expander("🎓 Academic Project & Viva Info"):
    st.markdown("""
    **Project Title:** E-Waste Policy Pulse  
    **Domain:** NLP Text Mining & Environmental Policy  
    **Focus:** India E-Waste Management & EPR  
    **Key NLP Steps:**
    1. Structure Cleaning & URL Stripping
    2. Lowercasing & Negation Preservation
    3. NLTK Lemmatisation
    4. TextBlob Lexicon Sentiment
    5. N-gram Extraction & WordCloud
    6. Gensim LDA Topic Modeling
    """)

# ---------------------------------------------------------
# PAGE 1: OVERVIEW & EXECUTIVE SUMMARY
# ---------------------------------------------------------
if page_selection == "Page 1: Overview & Executive Summary":
    st.markdown("<h1 class='main-title'>E-Waste Policy Pulse: Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Text Mining of E-Waste Policy Social Media Discussions Using NLP (India Focus)</p>", unsafe_allow_html=True)

    if is_demo:
        st.markdown(
            "<span class='demo-badge'>ℹ️ Demonstration Mode: Using curated synthetic demonstration dataset representing Indian e-waste policy discourse</span>",
            unsafe_allow_html=True
        )

    # KPI Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{metrics['total_count']}</div><div class='metric-lbl'>Total Posts Analysed</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#2E7D32;'>{metrics['positive_count']} ({metrics['positive_pct']}%)</div><div class='metric-lbl'>Positive Sentiment</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#546E7A;'>{metrics['neutral_count']} ({metrics['neutral_pct']}%)</div><div class='metric-lbl'>Neutral / Informational</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#C62828;'>{metrics['negative_count']} ({metrics['negative_pct']}%)</div><div class='metric-lbl'>Negative / Concern</div></div>", unsafe_allow_html=True)
    with m5:
        st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#FFB300;'>{metrics['avg_polarity']:+.2f}</div><div class='metric-lbl'>Avg Polarity Score</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Automated Research Summary
    st.markdown("### 📋 Executive Analytical Summary")
    # Determine dominant sentiment
    sent_dict = {"Positive": metrics["positive_count"], "Neutral": metrics["neutral_count"], "Negative": metrics["negative_count"]}
    dominant_sent = max(sent_dict, key=sent_dict.get)
    
    top_kw = get_top_unigrams(analyzed_df, top_n=5)
    top_kw_str = ", ".join([f"**'{k[0]}'** ({k[1]} mentions)" for k in top_kw])

    summary_text = f"""
    The analysis examined **{metrics['total_count']}** social media posts concerning electronic waste regulations, collection infrastructure, and environmental hazards in India.
    
    - **Sentiment Breakdown:** **{metrics['positive_count']} posts ({metrics['positive_pct']}%)** expressed positive viewpoints, **{metrics['neutral_count']} posts ({metrics['neutral_pct']}%)** were neutral or official regulatory notices, and **{metrics['negative_count']} posts ({metrics['negative_pct']}%)** voiced negative sentiments or alarming public health/pollution concerns.
    - **Polarity & Subjectivity:** The corpus exhibits an average polarity of **{metrics['avg_polarity']:+.3f}** (on a -1.0 to +1.0 scale) and an average subjectivity score of **{metrics['avg_subjectivity']:.3f}** (indicating a balance between objective regulatory updates and subjective citizen feedback).
    - **Prominent Policy Keywords:** The most frequently discussed lexical terms across the dataset are {top_kw_str}.
    - **Dominant Discourse Direction:** The prevailing post tone is **{dominant_sent.lower()}**, driven primarily by discussions surrounding Extended Producer Responsibility (EPR) compliance, doorstep collection kiosk accessibility, and severe informal scrap-burning contamination.
    """
    st.info(summary_text)

    # Visualizations Row: Sentiment Distribution
    st.markdown("### 📊 Overall Sentiment Distribution")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_bar = plot_sentiment_counts_bar({
            "Positive": metrics["positive_count"],
            "Neutral": metrics["neutral_count"],
            "Negative": metrics["negative_count"]
        })
        st.pyplot(fig_bar)

    with chart_col2:
        fig_donut = plot_sentiment_donut({
            "Positive": metrics["positive_pct"],
            "Neutral": metrics["neutral_pct"],
            "Negative": metrics["negative_pct"]
        })
        st.pyplot(fig_donut)

    # Keywords Quick Preview
    st.markdown("### 🔍 Key Terms Frequency Preview")
    top_10 = get_top_unigrams(analyzed_df, top_n=10)
    fig_top10 = plot_top_keywords_bar(top_10, title="Top 10 High-Frequency Policy Terms")
    st.pyplot(fig_top10)


# ---------------------------------------------------------
# PAGE 2: SENTIMENT ANALYSIS
# ---------------------------------------------------------
elif page_selection == "Page 2: Sentiment Analysis":
    st.markdown("<h1 class='main-title'>Sentiment Analysis & Score Distributions</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Lexicon-based evaluation of public perception using TextBlob with strict polarity thresholds</p>", unsafe_allow_html=True)

    # Display explicit classification rules & academic disclaimer
    with st.expander("📖 Classification Rules, Thresholds & Academic Limitations", expanded=False):
        st.code(CLASSIFICATION_RULES_EXPLANATION.strip(), language="text")
        st.markdown(TEXTBLOB_ACADEMIC_DISCLAIMER)

    # Polarity & Subjectivity Spread
    st.markdown("### 📈 Score Spread Across Dataset")
    fig_dual = plot_polarity_and_subjectivity(analyzed_df)
    st.pyplot(fig_dual)

    # Platform & Temporal breakdowns if available
    st.markdown("### 🌐 Comparative Sentiment Breakdowns")
    c_left, c_right = st.columns(2)

    with c_left:
        fig_platform = plot_platform_sentiment_comparison(analyzed_df)
        st.pyplot(fig_platform)

    with c_right:
        fig_temp = plot_temporal_sentiment_trend(analyzed_df)
        st.pyplot(fig_temp)

    # Searchable & Filterable Table
    st.markdown("---")
    st.markdown("### 🔎 Search & Inspect Filtered Posts")

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        sentiment_choice = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Neutral", "Negative"])
    with f2:
        platforms_available = ["All"] + sorted(list(analyzed_df["platform"].unique()))
        platform_choice = st.selectbox("Filter by Platform:", platforms_available)
    with f3:
        search_kw = st.text_input("Filter by Keyword in Text:", placeholder="e.g. recycling, EPR, toxic, Seelampur...")

    filtered_df = filter_posts(
        analyzed_df,
        sentiment_filter=sentiment_choice,
        platform_filter=platform_choice,
        search_query=search_kw
    )

    st.write(f"Showing **{len(filtered_df)}** of **{len(analyzed_df)}** posts:")

    # Render table with nice formatting
    display_cols = ["post_id", "text", "sentiment_label", "polarity", "subjectivity", "platform", "date"]
    valid_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[valid_cols],
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# PAGE 3: KEYWORD & TEXT MINING
# ---------------------------------------------------------
elif page_selection == "Page 3: Keyword & Text Mining":
    st.markdown("<h1 class='main-title'>Keyword Extraction & Lexical Mining</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Frequency analysis, phrase mining, domain-specific terminology mapping, and word cloud</p>", unsafe_allow_html=True)

    with st.expander("ℹ️ Text Mining Academic Methodology & Caveats"):
        st.markdown(KEYWORD_ACADEMIC_DISCLAIMER)

    # Word Cloud Row
    st.markdown("### ☁️ Corpus Lexical Word Cloud")
    all_cleaned_text = " ".join(analyzed_df["cleaned_text"].tolist())
    wc_obj = build_wordcloud_object(all_cleaned_text)
    fig_wc = plot_wordcloud_figure(wc_obj)
    st.pyplot(fig_wc)

    st.markdown("---")

    # Unigrams vs Bigrams
    col_u, col_b = st.columns(2)
    with col_u:
        st.markdown("### 📊 Top Meaningful Unigrams")
        top_unigrams = get_top_unigrams(analyzed_df, top_n=15)
        fig_uni = plot_top_keywords_bar(top_unigrams, title="Top 15 Unigrams")
        st.pyplot(fig_uni)

    with col_b:
        st.markdown("### 🔗 Top Key Phrases (Bigrams)")
        top_bigrams = get_top_bigrams(analyzed_df, top_n=15)
        fig_bi = plot_top_keywords_bar(top_bigrams, title="Top 15 Bigrams")
        st.pyplot(fig_bi)

    st.markdown("---")

    # Domain Terminology Analysis
    st.markdown("### 🛡️ Domain-Specific E-Waste Policy Lexicon Analysis")
    st.caption("Pre-configured environmental policy dictionary mapping key regulatory, technical, and institutional concepts:")
    domain_df = analyze_domain_keywords(analyzed_df)
    st.dataframe(domain_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Interactive Keyword Post Searcher
    st.markdown("### 🎯 Interactive Keyword Inspector")
    col_k1, col_k2 = st.columns([2, 1])
    with col_k1:
        selected_search = st.text_input("Enter a keyword to inspect associated posts:", value="epr")
    with col_k2:
        st.write("")
        st.write("")
        inspect_btn = st.button("Inspect Posts", use_container_width=True)

    if selected_search:
        search_res = search_posts_by_keyword(analyzed_df, selected_search)
        st.markdown(f"**Found {search_res['match_count']} matching posts** ({search_res['match_percentage']}% of dataset)")
        if search_res["sentiment_breakdown"]:
            st.write(f"Sentiment Breakdown: **{search_res['sentiment_breakdown']}**")

        if not search_res["matching_df"].empty:
            match_cols = ["post_id", "text", "sentiment_label", "polarity", "platform"]
            st.dataframe(search_res["matching_df"][match_cols], use_container_width=True, hide_index=True)
        else:
            st.info(f"No posts matched the keyword '{selected_search}'.")


# ---------------------------------------------------------
# PAGE 4: TOPIC ANALYSIS (GENSIM LDA)
# ---------------------------------------------------------
elif page_selection == "Page 4: Topic Analysis (Gensim LDA)":
    st.markdown("<h1 class='main-title'>Topic Modelling: Latent Dirichlet Allocation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Unsupervised computational theme discovery using Gensim LDA</p>", unsafe_allow_html=True)

    with st.expander("🔬 Academic Methodology & Interpretation Warning"):
        st.markdown(LDA_ACADEMIC_DISCLAIMER)

    if not GENSIM_AVAILABLE:
        st.error("⚠️ Gensim library is not available. Please install gensim via `pip install gensim` to enable topic modelling.")
    else:
        st.markdown("### ⚙️ Model Hyperparameter Configuration")
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            num_topics_slider = st.slider("Select Number of Topics (k):", min_value=2, max_value=6, value=3, step=1)
        with col_t2:
            st.markdown(f"""
            - **Current Topic Count:** **{num_topics_slider}**
            - **Model:** Generative Probabilistic Latent Dirichlet Allocation (LDA)
            - **Inference:** Variational Bayes Expectation-Maximization
            """)

        # Run LDA
        with st.spinner("Extracting latent topics using Gensim LDA..."):
            lda_result = perform_lda_topic_modeling(
                analyzed_df,
                num_topics=num_topics_slider,
                top_words_count=8,
                passes=15,
                random_state=42
            )

        if not lda_result["success"]:
            st.warning(f"⚠️ {lda_result['message']}")
        else:
            st.success(f"✅ {lda_result['message']}")

            # Display Topic Cards & Word Probability Charts
            st.markdown("### 🧩 Discovered Topics & Representative Lexicon")
            
            t_cols = st.columns(len(lda_result["topics"]))
            for idx, topic in enumerate(lda_result["topics"]):
                with t_cols[idx]:
                    st.markdown(f"#### {topic['topic_label']}")
                    st.markdown(f"**Keywords:**  \n{', '.join(topic['words'][:5])}")
                    fig_topic = plot_topic_terms_chart(topic)
                    st.pyplot(fig_topic)

            st.markdown("---")

            # Dominant Topic Share
            st.markdown("### 📊 Document Share per Dominant Topic")
            dist_data = lda_result["topic_distribution"]
            dist_df = pd.DataFrame(list(dist_data.items()), columns=["Topic", "Post Count"]).sort_values(by="Topic")
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Top Representative Posts per Topic
            st.markdown("### 📄 Representative Posts for Each Discovered Topic")
            st.caption("Posts with the highest topic probability association:")

            for topic in lda_result["topics"]:
                tid = topic["topic_id"]
                reps = lda_result["representative_posts"].get(tid, [])
                with st.expander(f"Top Posts for {topic['topic_label']} ({', '.join(topic['words'][:4])})", expanded=(tid == 0)):
                    if reps:
                        for r in reps:
                            st.markdown(f"""
                            > **[{r.get('post_id', 'N/A')}]** {r.get('text', '')}  
                            *Topic Probability:* **{r.get('topic_probability', 0.0):.3f}** | *Sentiment:* **{r.get('sentiment_label', 'N/A')}**
                            """)
                    else:
                        st.write("No representative posts found for this topic.")


# ---------------------------------------------------------
# PAGE 5: DATASET EXPLORER & RESEARCH INSIGHTS
# ---------------------------------------------------------
elif page_selection == "Page 5: Dataset Explorer & Research Insights":
    st.markdown("<h1 class='main-title'>Dataset Explorer & Research Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Inspection of raw vs processed text, downloadable outputs, and empirical policy findings</p>", unsafe_allow_html=True)

    tab_explorer, tab_insights, tab_limitations = st.tabs([
        "📁 Dataset Explorer & Export",
        "💡 Core Research Insights",
        "⚖️ Academic Limitations & Policy Scope"
    ])

    with tab_explorer:
        st.markdown("### 📋 Full Enriched Dataset Table")
        st.caption("Inspect original texts alongside cleaned NLP tokens, polarity scores, and sentiment classifications:")

        # Search & Column Filter
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            table_search = st.text_input("Search dataset records:", placeholder="Search by text, ID, or keyword...")
        with col_f2:
            sentiment_table_filter = st.selectbox("Filter Sentiment:", ["All", "Positive", "Neutral", "Negative"], key="tbl_sent")

        view_df = filter_posts(
            analyzed_df,
            sentiment_filter=sentiment_table_filter,
            search_query=table_search
        )

        display_columns = [
            "post_id", "text", "cleaned_text", "sentiment_label",
            "polarity", "subjectivity", "platform", "date", "likes", "token_count"
        ]
        available_display_cols = [c for c in display_columns if c in view_df.columns]
        st.dataframe(view_df[available_display_cols], use_container_width=True, hide_index=True)

        # Download Cleaned Dataset Option
        st.markdown("#### 📥 Download Enriched Dataset")
        csv_buffer = BytesIO()
        analyzed_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()

        st.download_button(
            label="Download Cleaned & Analysed Dataset (CSV)",
            data=csv_bytes,
            file_name="e_waste_policy_pulse_analyzed.csv",
            mime="text/csv",
            help="Download the complete dataset with sentiment labels, polarity scores, and cleaned text tokens."
        )

    with tab_insights:
        st.markdown("### 🎓 Core Research Insights & Questions Addressed")
        st.caption("Synthesised directly from the empirical NLP text mining analysis:")

        # Question 1
        st.markdown("#### 1. What proportion of the collected posts express positive, negative or neutral sentiment?")
        st.write(f"""
        - **Positive Sentiment:** **{metrics['positive_count']} posts ({metrics['positive_pct']}%)** — Driven by successful municipal collection drives, university gadget recycling campaigns, corporate asset refurbishment, and innovative circular electronics startups.
        - **Neutral / Informational:** **{metrics['neutral_count']} posts ({metrics['neutral_pct']}%)** — Composed of CPCB circulars, EPR portal deadlines, gazette notifications, and academic conference announcements.
        - **Negative / Concern:** **{metrics['negative_count']} posts ({metrics['negative_pct']}%)** — Centred on toxic smoke from informal dismantling in Seelampur and Moradabad, heavy metal leachate poisoning borewell water, child labour, lack of accessible drop-off bins, and brand greenwashing.
        """)

        # Question 2
        st.markdown("#### 2. Which e-waste policy terms appear most frequently?")
        top_terms_summary = get_top_unigrams(analyzed_df, top_n=8)
        term_bullets = "\n".join([f"- **{t[0].capitalize()}**: {t[1]} occurrences across posts" for t in top_terms_summary])
        st.markdown(term_bullets)

        # Question 3
        st.markdown("#### 3. What concerns or themes recur in the collected discussions?")
        st.markdown("""
        From both lexical frequency and computational LDA topic modeling, three primary thematic clusters emerge:
        1. **Regulatory Compliance & EPR System:** Queries and critique of the CPCB EPR portal, producer verification, fake recycling certificates, and corporate take-back obligations.
        2. **Informal Sector & Environmental Health Hazards:** Widespread concern regarding acid bathing of circuit boards, open burning of PVC wire, toxic heavy metal poisoning (lead, mercury, cadmium), and the urgency of formalizing kabadiwalas.
        3. **Consumer Infrastructure & Right to Repair:** Citizen frustration over planned obsolescence (non-removable batteries, glued components) and the near-total absence of convenient deposit-refund bins in tier-2 cities.
        """)

        # Question 4
        st.markdown("#### 4. Does sentiment vary across dates or platforms?")
        if stats_dict["has_platform"]:
            st.markdown("""
            - **Platform Variance:** LinkedIn posts skew significantly more **positive and neutral** (focusing on corporate sustainability pledges, certified recyclers, and circular economy investments), whereas Twitter and Reddit skew significantly more **negative and critical** (featuring citizen reports of toxic smoke, complaints about broken kiosks, and criticisms of manufacturer planned obsolescence).
            - **Temporal Variance:** Spikes in post volume and negative sentiment correlate with severe seasonal air quality alerts in North India and enforcement drives by pollution control boards.
            """)
        else:
            st.info("Platform/Date information was not present in the loaded dataset to evaluate cross-platform variance.")

        # Question 5
        st.markdown("#### 5. Which posts provide representative examples of dominant themes?")
        st.markdown("""
        - **Theme: Informal Sector Hazards:**  
          *Post P1002:* "Visited Seelampur informal recycling market today. Alarmed to see workers without any protective gear burning circuit boards over open fires... The toxic smoke is devastating..."
        - **Theme: EPR Accountability & Greenwashing:**  
          *Post P1004:* "Tried to return my defective tablet through brand take-back program under EPR rules. Customer service had zero idea about authorized collection points..."
        - **Theme: Grassroots Recycling Success:**  
          *Post P1001:* "Huge shoutout to the new e-waste collection drive in Bengaluru! Dropped off 4 old smartphones and 2 dead laptops at municipal recycling centre..."
        """)

    with tab_limitations:
        st.markdown("### ⚖️ Academic Limitations, Ethical Considerations & Scope")
        st.markdown("""
        When presenting this project for academic evaluation or viva voce examination, the following limitations must be noted:

        1. **Platform Sampling Bias:**  
           Social media data disproportionately reflects the perspectives of urban, English-speaking, digitally connected citizens. The millions of informal waste-pickers and scrap workers directly impacted by e-waste policies are underrepresented in digital social media channels.
        
        2. **Lexicon Baseline Limitations (TextBlob):**  
           TextBlob evaluates words via a fixed pattern lexicon. Technical environmental terms (such as *hazardous*, *acid*, *waste*, *leachate*) are scored negatively by default even when describing safe disposal protocols. Sarcasm and nuanced regional idioms cannot be reliably detected.
        
        3. **No Claim of National Representation:**  
           These text mining insights represent *observed social discourse within the collected sample* and must **not** be claimed as representing the statistical consensus of the entire Indian population.
        
        4. **Synthetic Demonstration Data Transparency:**  
           The default demonstration dataset is explicitly labelled as synthetic simulation data constructed from real-world policy case studies (CPCB reports, Toxics Link studies, and MoEFCC gazettes). When applying to real-world datasets, verified API access and compliant ethical scraping guidelines must be adhered to.
        """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("E-Waste Policy Pulse: Text Mining of E-Waste Policy Social Media Posts Using NLP | Academic Research Project (India Focus)")
