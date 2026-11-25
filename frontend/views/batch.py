"""Tab 2: upload a CSV of reviews and score them all with each model."""

from pathlib import Path

import pandas as pd
import streamlit as st

from backend.config import MAX_BATCH_ROWS, MODEL_SPECS
from controllers import insights_controller as ic
from controllers import prediction_controller as pc
from frontend import charts
from frontend import components as ui

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "assets" / "sample_reviews.csv"


def render() -> None:
    ui.section("Batch analysis",
               f"Upload a CSV with one review per row (up to {MAX_BATCH_ROWS} rows). Useful for a brand or "
               "category manager who wants to triage many reviews at once and spot the negative ones.",
               kicker="Many reviews at once")

    c1, c2 = st.columns([3, 1.2], gap="large")
    with c1:
        upload = st.file_uploader("Reviews CSV", type=["csv"],
                                  help="Needs a column with the review text. A title/summary column is optional.")
    with c2:
        st.write("")
        if st.button("Use sample file (12 reviews)", width="stretch"):
            st.session_state["batch_source"] = "sample"
            st.session_state.pop("batch_result", None)
        st.download_button("Download CSV template", SAMPLE_CSV.read_bytes(), "sample_reviews.csv",
                           "text/csv", width="stretch")

    if upload is not None:
        st.session_state["batch_source"] = "upload"
    source = st.session_state.get("batch_source")
    if source is None or (source == "upload" and upload is None):
        ui.alert("info", "ℹ", "Upload a CSV or use the sample file to get started.",
                 "Example columns: <code>Summary</code>, <code>Text</code> (same as the Kaggle dataset).")
        return

    try:
        df = pd.read_csv(upload if source == "upload" else SAMPLE_CSV)
    except Exception as exc:
        ui.alert("warn", "⚠", "Could not read that file as CSV.", str(exc))
        return
    if df.empty:
        ui.alert("warn", "⚠", "The file has no rows.")
        return

    columns = list(df.columns)
    text_guess = pc.guess_column(columns, ["text", "review", "review_text", "reviewtext", "body", "content"])
    summary_guess = pc.guess_column(columns, ["summary", "title", "headline"])

    s1, s2, s3 = st.columns([1, 1, 1.2])
    text_col = s1.selectbox("Review text column", columns,
                            index=columns.index(text_guess) if text_guess else 0)
    summary_options = ["(none)"] + columns
    summary_col = s2.selectbox("Title / summary column", summary_options,
                               index=summary_options.index(summary_guess) if summary_guess else 0)
    available = [s.key for s in pc.available_specs()]
    chosen = s3.multiselect("Models", [s.key for s in MODEL_SPECS], default=available,
                            format_func=lambda k: pc.SPECS_BY_KEY[k].name)

    if len(df) > MAX_BATCH_ROWS:
        st.caption(f"Only the first {MAX_BATCH_ROWS} of {len(df):,} rows will be analyzed.")

    if st.button(f"Analyze {min(len(df), MAX_BATCH_ROWS)} reviews", type="primary"):
        if not chosen:
            ui.alert("warn", "⚠", "Select at least one model.")
            return
        bar = st.progress(0.0, "Running models…")
        result = pc.analyze_batch(df, text_col, None if summary_col == "(none)" else summary_col, chosen,
                                  progress=lambda frac, msg: bar.progress(frac, msg))
        bar.empty()
        st.session_state["batch_result"] = (result, chosen)

    if "batch_result" not in st.session_state:
        return
    result, chosen = st.session_state["batch_result"]
    _render_batch_results(result, chosen)


def _render_batch_results(result: pd.DataFrame, chosen: list[str]) -> None:
    agreement = pc.batch_agreement(result)
    tiles = [(f"{len(result):,}", "reviews analyzed")]
    for key in chosen:
        col = f"{pc.SPECS_BY_KEY[key].name} label"
        if col in result:
            share = (result[col] == "Negative").mean()
            tiles.append((f"{share:.0%}", f"flagged Negative by {pc.SPECS_BY_KEY[key].name}"))
    if agreement is not None:
        tiles.append((f"{agreement:.0%}", "model agreement"))
    ui.stat_row(tiles)

    dist = pc.batch_distribution(result, chosen)
    if not dist.empty:
        st.write("")
        st.altair_chart(charts.batch_distribution(dist, ic.model_colors()), width="stretch")

    label_cols = [c for c in result.columns if c.endswith(" label")]
    only_disagree = st.toggle("Show only reviews where the models disagree", value=False,
                              disabled=len(label_cols) < 2)
    view = result[result[label_cols].nunique(axis=1) > 1] if only_disagree and len(label_cols) > 1 else result
    conf_cols = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="percent")
                 for c in result.columns if c.endswith(" confidence")}
    st.dataframe(view, hide_index=True, width="stretch", column_config=conf_cols)
    st.download_button("Download results CSV", result.to_csv(index=False).encode("utf-8"),
                       "sentiment_predictions.csv", "text/csv", type="primary")
