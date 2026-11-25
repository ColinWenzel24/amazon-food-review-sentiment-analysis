"""Tab 4: the data behind the models."""

import streamlit as st

from controllers import insights_controller as ic
from frontend import charts
from frontend import components as ui


def render() -> None:
    ui.section("The data",
               "A shared 50,000-review sample (seed 48) of the Amazon Fine Food Reviews dataset "
               "(568,454 reviews, 1999–2012). Star ratings become sentiment labels: "
               "1–2★ Negative, 3★ Neutral, 4–5★ Positive.",
               kicker="Dataset overview")

    s = ic.dataset_summary()
    if s is None:
        ui.alert("info", "ℹ", "Dataset summary not generated yet.",
                 "Run <code>python scripts/export_classical_models.py</code> to create "
                 "<code>results/dataset_summary.json</code>.")
        return

    ui.stat_row([
        (f"{s['raw_rows']:,}", "reviews in sample"),
        (f"{s['clean_rows']:,}", "after cleaning"),
        (f"{s['train_rows']:,} / {s['test_rows']:,}", "train / test (80/20)"),
        (f"{s['avg_words']:.0f}", "average words per review"),
        (f"{s['vocabulary_size']:,}", "TF-IDF vocabulary"),
    ])
    st.write("")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Star ratings**")
        st.altair_chart(charts.star_distribution(s["score_counts"]), width="stretch")
    with c2:
        st.markdown("**Sentiment labels after mapping**")
        st.altair_chart(charts.sentiment_distribution(s["class_counts"]), width="stretch")
    pos = s["class_counts"]["Positive"] / s["clean_rows"]
    st.caption(f"Strong class imbalance: {pos:.0%} of reviews are Positive. A model that always says "
               "“Positive” is already right 78% of the time, so we judge models by macro-F1, and DistilBERT "
               "uses class weights during training.")

    c3, c4 = st.columns([1.2, 1], gap="large")
    with c3:
        st.markdown("**Review length**")
        st.altair_chart(charts.length_histogram(s["length_histogram"]), width="stretch")
        med = s["median_words_by_class"]
        st.caption(f"Median length: {s['median_words']:.0f} words (Negative {med['Negative']:.0f}, "
                   f"Neutral {med['Neutral']:.0f}, Positive {med['Positive']:.0f}). About 6% of reviews exceed "
                   "DistilBERT's 256-token limit and are truncated.")
    with c4:
        st.markdown("**Cleaning steps**")
        ui.html(f"""
        <div class="card">
          <ol style="margin:0; padding-left:18px; line-height:1.7;">
            <li>Dropped <b>{s['missing_summary']}</b> reviews with no summary/title.</li>
            <li>Removed <b>{s['duplicate_texts_removed']:,}</b> duplicate review texts (the same review posted
                under several product variants) so no review appears in both train and test.</li>
            <li>Stripped HTML tags such as <code>&lt;br /&gt;</code> (found in
                <b>{s['html_reviews']:,}</b> reviews) and extra whitespace.</li>
            <li>Joined the <b>title + review text</b> into one input for every model.</li>
          </ol>
        </div>""")

    words = ic.top_words()
    if words:
        st.markdown("##### Most informative words (Logistic Regression weights)")
        w1, w2 = st.columns(2, gap="large")
        w1.altair_chart(charts.top_words_chart(words["Positive"], "Positive"), width="stretch")
        w2.altair_chart(charts.top_words_chart(words["Negative"], "Negative"), width="stretch")
        st.caption("“not” is the strongest negative word, but a bag-of-words model cannot tell “not good” "
                   "from “good, not bad”. That is one reason the context-aware DistilBERT does better.")
