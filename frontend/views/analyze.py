"""Tab 1: type or paste a review and compare live predictions from each model side by side."""

from html import escape

import streamlit as st

from backend.config import MAX_REVIEW_CHARS, MODEL_SPECS
from controllers import prediction_controller as pc
from frontend import components as ui

EXAMPLES = {
    "😊 Positive": ("Best coffee I've had in years",
                   "Smooth, rich and never bitter. I ordered a second bag before the first one was even "
                   "finished. Highly recommend!"),
    "😞 Negative": ("Stale and tasteless",
                   "The chips arrived crushed and tasted stale. I threw the whole bag away and will not be "
                   "ordering again."),
    "😐 Mixed / neutral": ("Okay, but overpriced",
                          "The flavor is decent and my kids ate them, but for this price I expected better "
                          "quality. It's fine, nothing special. Probably won't reorder."),
    "🤔 Tricky (sarcasm)": ("Wow, just great",
                           "Oh great, another 'gourmet' snack that tastes like cardboard. Five stars for the "
                           "packaging, I guess."),
}


def _load_example(name: str) -> None:
    st.session_state["review_summary"], st.session_state["review_text"] = EXAMPLES[name]
    st.session_state.pop("analysis", None)


def _clear() -> None:
    st.session_state["review_summary"] = ""
    st.session_state["review_text"] = ""
    st.session_state.pop("analysis", None)


def render() -> None:
    ui.section("Analyze a review",
               "Type or paste a food review. Each model predicts whether it is Negative (1–2★), "
               "Neutral (3★), or Positive (4–5★), and the results appear side by side.",
               kicker="Live prediction")

    st.write("**Try an example:**")
    cols = st.columns(len(EXAMPLES) + 1)
    for col, name in zip(cols, EXAMPLES):
        col.button(name, on_click=_load_example, args=(name,), width="stretch")
    cols[-1].button("✕ Clear", on_click=_clear, width="stretch")

    left, right = st.columns([2.3, 1.2], gap="large")
    with left:
        summary = st.text_input("Review title (optional)", key="review_summary",
                                placeholder="e.g. Delicious and fresh",
                                help="The models were trained on title + review text, so adding a title "
                                     "usually helps.")
        text = st.text_area("Review text", key="review_text", height=150, max_chars=MAX_REVIEW_CHARS,
                            placeholder="Write what you thought of the product…")
    with right:
        status = pc.model_status()
        options = [s.key for s in MODEL_SPECS]
        default = [k for k in options if status[k] is None] or options
        chosen = st.multiselect("Models to compare", options, default=default,
                                format_func=lambda k: pc.SPECS_BY_KEY[k].name)
        run = st.button("Analyze sentiment", type="primary", width="stretch")
        st.caption("DistilBERT runs on CPU in the cloud. A prediction usually takes under a second.")

    if run:
        error = pc.validate_review(text)
        if error:
            ui.alert("warn", "⚠", error)
            return
        if not chosen:
            ui.alert("warn", "⚠", "Select at least one model.")
            return
        with st.spinner("Reading the review…"):
            st.session_state["analysis"] = pc.analyze_review(text, summary, chosen)

    analysis = st.session_state.get("analysis")
    if not analysis:
        return

    _render_results(analysis)


def _render_results(analysis: "pc.AnalysisResult") -> None:
    st.write("")
    ok = analysis.successful
    if analysis.agreement is True:
        label = ok[0].prediction.label
        ui.alert("ok", "✔", f"All {len(ok)} models agree: {label}",
                 "When models built in very different ways agree, the prediction is more trustworthy.")
    elif analysis.agreement is False:
        verdicts = ", ".join(f"{r.spec.name} → {r.prediction.label}" for r in ok)
        ui.alert("warn", "⚖", "The models disagree",
                 f"{escape(verdicts)}. Disagreement often signals mixed opinions, sarcasm, or negation "
                 "that word-count features miss.")

    cols = st.columns(len(analysis.results), gap="medium")
    for col, result in zip(cols, analysis.results):
        with col:
            ui.prediction_card(result)

    ui.html('<div style="height: 28px;"></div>')  # breathing room so the expander isn't glued to the cards
    with st.expander("What the models actually read"):
        st.caption("HTML tags removed, whitespace collapsed, and the title joined to the review text, "
                   "exactly as in training.")
        st.code(analysis.model_input, language=None, wrap_lines=True)
