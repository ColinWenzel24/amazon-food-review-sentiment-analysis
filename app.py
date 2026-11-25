"""Streamlit entry point:  streamlit run app.py

Layout of the code:
    backend/      preprocessing, model loading, inference, reading saved results (no Streamlit)
    controllers/  validates input, caches models, prepares data for the pages
    frontend/     theme, reusable components, charts, and one module per tab (frontend/views/)
"""

import streamlit as st

st.set_page_config(
    page_title="reviewsense · Food Review Sentiment",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from backend.config import MODEL_SPECS  # noqa: E402
from controllers import prediction_controller as pc  # noqa: E402
from frontend import components as ui  # noqa: E402
from frontend.theme import apply_theme  # noqa: E402
from frontend.views import about, analyze, batch, compare, dataset  # noqa: E402


def main() -> None:
    apply_theme()
    with st.spinner("Loading models… (first start downloads DistilBERT, about 270 MB)"):
        status = pc.model_status()
    ui.navbar(sum(err is None for err in status.values()), len(MODEL_SPECS))

    tabs = st.tabs(["🔍 Analyze a review", "📦 Batch analysis", "📊 Model comparison", "🗂️ The data",
                    "ℹ️ How it works"])
    with tabs[0]:
        analyze.render()
    with tabs[1]:
        batch.render()
    with tabs[2]:
        compare.render()
    with tabs[3]:
        dataset.render()
    with tabs[4]:
        about.render()

    ui.footer()


main()
