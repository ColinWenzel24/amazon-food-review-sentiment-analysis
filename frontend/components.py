"""Reusable HTML components. All user-provided text is escaped before rendering."""

from html import escape

import streamlit as st

from backend.config import LABELS, STAR_RANGE
from frontend.theme import SENTIMENT_STYLE


def html(markup: str) -> None:
    # Flatten to one line: indented or blank lines would otherwise be parsed as Markdown code blocks
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def navbar(models_ready: int, models_total: int) -> None:
    html(f"""
    <div class="nav bleed">
      <div>
        <div class="brand">review<span class="accent">sense</span><span class="underline"></span></div>
        <div class="tagline">Food review sentiment intelligence</div>
      </div>
      <div style="display:flex; gap:10px;">
        <div class="pill">Models online<b>{models_ready} of {models_total}</b></div>
        <div class="pill">Trained on<b>46,261 reviews</b></div>
        <div class="pill">Team<b>Group Psi · CSE 3521</b></div>
      </div>
    </div>
    <div class="subnav bleed">
      <span><span class="dot">●</span> Paste a review, get a sentiment</span>
      <span>Logistic Regression + TF-IDF</span>
      <span>DistilBERT transformer</span>
      <span>Negative · Neutral · Positive</span>
    </div>
    """)


def section(title: str, subtitle: str = "", kicker: str = "") -> None:
    k = f'<div class="kicker">{kicker}</div>' if kicker else ""
    s = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    html(f'{k}<div class="section-title">{title}</div>{s}')


def stars_for(label: str) -> str:
    filled = {"Negative": 1.5, "Neutral": 3, "Positive": 4.5}[label]
    full = int(filled)
    half = filled - full >= 0.5
    return "★" * full + ("⯪" if half else "") + "☆" * (5 - full - (1 if half else 0))


def badge(label: str) -> str:
    s = SENTIMENT_STYLE[label]
    return (f'<span class="badge" style="color:{s["color"]}; background:{s["bg"]};">'
            f'{s["icon"]} {label}</span>')


def probability_rows(probabilities: dict[str, float], predicted: str) -> str:
    rows = []
    for label in reversed(LABELS):  # Positive on top, like a review histogram
        p = probabilities[label]
        top = " top" if label == predicted else ""
        rows.append(
            f'<div class="hist-row{top}"><span class="lbl">{label}</span>'
            f'<div class="track"><div class="fill" style="width:{p * 100:.1f}%"></div></div>'
            f'<span class="pct">{p * 100:.0f}%</span></div>'
        )
    return "".join(rows)


def word_chips(words: list[tuple[str, float]], kind: str) -> str:
    if not words:
        return '<span style="color:#565959;font-size:.82rem;">none</span>'
    return '<div class="chips">' + "".join(
        f'<span class="chip {kind}" title="contribution {w:+.3f}">{escape(word)}</span>' for word, w in words
    ) + "</div>"


def prediction_card(result) -> None:
    """Render one model's prediction (controllers.prediction_controller.ModelResult)."""
    spec, pred = result.spec, result.prediction
    family = "Classical ML" if spec.family == "classical" else "Deep learning"
    head = (f'<h4>{escape(spec.name)}</h4>'
            f'<div class="meta">{family} · {escape(spec.features)}</div>')

    if pred is None:
        html(f'<div class="card">{head}<div class="alert warn"><span class="icon">⚠</span>'
             f'<div><b>Model unavailable</b><br><span style="font-size:.85rem">{escape(result.error or "")}'
             f'</span></div></div></div>')
        return

    explanation = ""
    if pred.supporting_words or pred.opposing_words:
        explanation = (
            f'<div style="margin-top:14px;font-size:.85rem;font-weight:700;">Words pushing toward '
            f'{pred.label}</div>{word_chips(pred.supporting_words, "for")}'
            f'<div style="margin-top:10px;font-size:.85rem;font-weight:700;">Words pushing away</div>'
            f'{word_chips(pred.opposing_words, "against")}'
        )
    elif spec.family == "deep":
        explanation = ('<div style="margin-top:14px;font-size:.82rem;color:#565959;">DistilBERT reads the whole '
                       'sentence in context (word order, negation like "not good"), so there is no simple '
                       'per-word weight to show.</div>')
    truncated = ('<div style="margin-top:8px;font-size:.8rem;color:#C45500;">Long review: only the first 256 '
                 'tokens were read by the model.</div>') if pred.truncated else ""

    html(f"""
    <div class="card">
      {head}
      <div>{badge(pred.label)}<span class="conf">{pred.confidence * 100:.1f}% confident</span></div>
      <div style="margin:8px 0 4px;font-size:.85rem;color:#565959;">
        Looks like a <span class="stars">{stars_for(pred.label)}</span> review ({STAR_RANGE[pred.label]})
      </div>
      {probability_rows(pred.probabilities, pred.label)}
      {explanation}
      {truncated}
      <div style="margin-top:12px;font-size:.75rem;color:#565959;">Inference time: {pred.latency_ms:.0f} ms</div>
    </div>
    """)


def alert(kind: str, icon: str, title: str, body: str = "") -> None:
    html(f'<div class="alert {kind}"><span class="icon">{icon}</span>'
         f'<div><b>{title}</b>{"<br>" + body if body else ""}</div></div>')


def stat_tile(value: str, label: str) -> str:
    return f'<div class="stat"><div class="v">{value}</div><div class="k">{label}</div></div>'


def stat_row(tiles: list[tuple[str, str]]) -> None:
    cols = st.columns(len(tiles))
    for col, (value, label) in zip(cols, tiles):
        with col:
            html(stat_tile(value, label))


def footer() -> None:
    html("""
    <div class="footer bleed">
      CSE 3521 · Survey of Artificial Intelligence I · The Ohio State University · Group Psi<br>
      Data: <a href="https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews" target="_blank">
      Amazon Fine Food Reviews (SNAP / Kaggle)</a>. Student project for educational use; not affiliated with Amazon.
    </div>
    """)
