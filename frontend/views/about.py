"""Tab 5: how the system works, for non-technical users and for future engineers."""

import streamlit as st

from backend.config import distilbert_source
from controllers import prediction_controller as pc
from frontend import components as ui

TEAM = [
    ("Abhinav Siriguri", "Project Lead"),
    ("Colin Wenzel", "Data Engineer: cleaning, feature engineering (BoW, TF-IDF)"),
    ("Karan Tandra", "ML Engineer: Logistic Regression, baseline, error analysis"),
    ("KC Luna", "ML Engineer: DistilBERT fine-tuning"),
    ("Ardelia Putridaryana", "App Engineer: Streamlit app and deployment"),
    ("Jackson Ball", "Analyst / Writer"),
]


def render() -> None:
    ui.section("How it works", "From a raw review to a sentiment prediction.", kicker="About")

    steps = [
        ("1", "You type a review", "Optionally with a title, just like on a product page."),
        ("2", "Clean the text", "Remove HTML tags and extra spaces, then join title + review."),
        ("3", "Turn words into numbers",
         "TF-IDF scores each word by how distinctive it is. DistilBERT splits text into word pieces and "
         "builds context-aware vectors."),
        ("4", "Predict", "Each model outputs a probability for Negative, Neutral, and Positive."),
    ]
    cols = st.columns(len(steps))
    for col, (n, title, body) in zip(cols, steps):
        with col:
            ui.html(f'<div class="card"><div class="kicker">Step {n}</div><h4>{title}</h4>'
                    f'<div style="font-size:.88rem;color:#565959;margin-top:6px;">{body}</div></div>')

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### The models")
        st.markdown(
            "- **Logistic Regression + TF-IDF** (classical): fast and transparent. Every word has a weight, "
            "so the app can show which words drove the decision. It treats text as a bag of words and "
            "ignores word order.\n"
            "- **DistilBERT** (deep learning): a smaller, faster version of BERT pre-trained on English text "
            "and fine-tuned on our reviews. It reads words in context, so it handles negation "
            "(“not good”) and mixed reviews better, at a higher computational cost.\n"
            "- **Majority baseline**: always predicts “Positive”. Any useful model must beat it.")
        st.markdown("##### Who is this for?")
        st.markdown(
            "Brand and category managers, marketplace trust & safety teams, and product researchers who need "
            "to triage thousands of reviews: flag unhappy customers quickly, track sentiment for a product, "
            "or catch reviews whose star rating doesn't match their text.")
    with c2:
        st.markdown("##### Using the app")
        st.markdown(
            "1. **Analyze a review**: paste a review (or click an example), choose models, and press "
            "*Analyze sentiment*. Predictions appear side by side with confidence bars.\n"
            "2. **Batch analysis**: upload a CSV to score many reviews and download the results.\n"
            "3. **Model comparison**: test-set accuracy, precision, recall, F1, training time, confusion "
            "matrices, and DistilBERT's training curves.\n"
            "4. **The data**: what the training data looks like and how it was cleaned.")
        st.markdown("##### Limitations")
        st.markdown(
            "Predictions reflect 2000s-era Amazon food reviews. Neutral (3★) reviews are hard for every "
            "model, and sarcasm can fool both. Treat low-confidence predictions with care.")

    st.markdown("##### Team: Group Psi")
    tcols = st.columns(3)
    for i, (name, role) in enumerate(TEAM):
        tcols[i % 3].markdown(f"**{name}**  \n<span style='color:#565959;font-size:.88rem'>{role}</span>",
                              unsafe_allow_html=True)

    with st.expander("Technical details"):
        status = pc.model_status()
        for key, err in status.items():
            st.write(f"- **{pc.SPECS_BY_KEY[key].name}**: {'✅ loaded' if err is None else '⚠ ' + err}")
        st.write(f"- DistilBERT weights source: `{distilbert_source()}`")
        st.write("- Code: `backend/` (preprocessing, model loading, inference) · `controllers/` "
                 "(input validation, caching, orchestration) · `frontend/` (pages, components, charts, theme).")
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
