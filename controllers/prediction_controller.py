"""Live prediction: single reviews and CSV batches."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from backend.config import LABELS, MAX_BATCH_ROWS, MAX_REVIEW_CHARS, MODEL_SPECS, ModelSpec
from backend.models import ModelUnavailableError, Prediction, load_model
from backend.preprocessing import combine_review

SPECS_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


@st.cache_resource(show_spinner=False)
def _get_model(key: str):
    """Load a model once per server process. Errors are returned, not raised, so they are cached too."""
    try:
        return load_model(SPECS_BY_KEY[key]), None
    except ModelUnavailableError as exc:
        return None, str(exc)


def model_status() -> dict[str, str | None]:
    """{model_key: None if ready, else an error message}. Loads (and caches) every model."""
    return {spec.key: _get_model(spec.key)[1] for spec in MODEL_SPECS}


def available_specs() -> list[ModelSpec]:
    return [spec for spec in MODEL_SPECS if _get_model(spec.key)[0] is not None]


@dataclass
class ModelResult:
    spec: ModelSpec
    prediction: Prediction | None
    error: str | None = None


@dataclass
class AnalysisResult:
    model_input: str
    results: list[ModelResult]

    @property
    def successful(self) -> list[ModelResult]:
        return [r for r in self.results if r.prediction is not None]

    @property
    def agreement(self) -> bool | None:
        labels = {r.prediction.label for r in self.successful}
        return None if len(self.successful) < 2 else len(labels) == 1


def validate_review(text: str) -> str | None:
    """Return an error message, or None if the review is usable."""
    if not text or not text.strip():
        return "Please enter a review to analyze."
    if len(text.strip().split()) < 2:
        return "Please enter at least a couple of words; single words give unreliable predictions."
    if len(text) > MAX_REVIEW_CHARS:
        return f"Reviews are limited to {MAX_REVIEW_CHARS:,} characters (yours has {len(text):,})."
    return None


def analyze_review(text: str, summary: str, model_keys: list[str]) -> AnalysisResult:
    model_input = combine_review(summary, text)
    results = []
    for key in model_keys:
        spec = SPECS_BY_KEY[key]
        model, error = _get_model(key)
        if model is None:
            results.append(ModelResult(spec, None, error))
            continue
        try:
            results.append(ModelResult(spec, model.predict([model_input])[0]))
        except Exception as exc:  # keep the other models' results if one fails
            results.append(ModelResult(spec, None, f"Prediction failed: {exc}"))
    return AnalysisResult(model_input, results)


# ---------------------------------------------------------------- batch

def guess_column(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def analyze_batch(df: pd.DataFrame, text_col: str, summary_col: str | None,
                  model_keys: list[str], progress=None) -> pd.DataFrame:
    """Run every selected model on up to MAX_BATCH_ROWS reviews. Returns one row per review."""
    df = df.head(MAX_BATCH_ROWS).copy()
    df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]
    summaries = df[summary_col].fillna("").astype(str) if summary_col else [""] * len(df)
    inputs = [combine_review(s, t) for s, t in zip(summaries, df[text_col].astype(str))]

    out = pd.DataFrame({"review": df[text_col].astype(str).str.slice(0, 300).values})
    if summary_col:
        out.insert(0, "summary", df[summary_col].fillna("").astype(str).values)

    for n, key in enumerate(model_keys):
        spec = SPECS_BY_KEY[key]
        model, error = _get_model(key)
        if model is None:
            continue
        preds = model.predict(inputs, explain=False)
        out[f"{spec.name} label"] = [p.label for p in preds]
        out[f"{spec.name} confidence"] = [round(p.confidence, 4) for p in preds]
        if progress:
            progress((n + 1) / len(model_keys), f"{spec.name} done")
    return out


def batch_distribution(result_df: pd.DataFrame, model_keys: list[str]) -> pd.DataFrame:
    """Long-format counts: model, sentiment, count (for the batch chart)."""
    rows = []
    for key in model_keys:
        col = f"{SPECS_BY_KEY[key].name} label"
        if col not in result_df:
            continue
        counts = result_df[col].value_counts()
        for label in LABELS:
            rows.append({"Model": SPECS_BY_KEY[key].name, "Sentiment": label, "Reviews": int(counts.get(label, 0))})
    return pd.DataFrame(rows)


def batch_agreement(result_df: pd.DataFrame) -> float | None:
    label_cols = [c for c in result_df.columns if c.endswith(" label")]
    if len(label_cols) < 2 or result_df.empty:
        return None
    return float((result_df[label_cols].nunique(axis=1) == 1).mean())
