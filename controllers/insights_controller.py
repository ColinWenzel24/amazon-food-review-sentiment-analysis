"""Model comparison and dataset insights, built from the saved results in results/."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend import results
from backend.config import LABELS, MODEL_SPECS, ModelSpec

BASELINE_NAME = "Majority baseline"
BASELINE_COLOR = "#8D9096"


def _format_time(minutes: float | None, device: str | None) -> str:
    if minutes is None:
        return "—"
    label = f"{minutes * 60:.1f} s" if minutes < 1 else f"{minutes:.1f} min"
    if device and device != "cpu":
        return f"{label} (GPU)"
    return f"{label} (CPU)" if device == "cpu" else label


@st.cache_data(show_spinner=False)
def comparison_table() -> pd.DataFrame:
    """One row per model: accuracy, macro precision/recall/F1, weighted F1, training time."""
    rows = []
    baseline = results.load_baseline_metrics()
    if baseline:
        rows.append(_row(BASELINE_NAME, "reference", "—", baseline))
    for spec in MODEL_SPECS:
        m = results.load_metrics(spec.metrics_file)
        if m:
            rows.append(_row(spec.name, spec.family, spec.features, m))
    return pd.DataFrame(rows)


def _row(name: str, family: str, features: str, m: dict) -> dict:
    return {
        "Model": name,
        "Type": family.capitalize(),
        "Accuracy": m["accuracy"],
        "Precision (macro)": m["macro_precision"],
        "Recall (macro)": m["macro_recall"],
        "F1 (macro)": m["macro_f1"],
        "F1 (weighted)": m.get("weighted_f1"),
        "Training time": _format_time(m.get("training_minutes"), m.get("device")),
        "Features": features,
    }


def missing_metrics() -> list[ModelSpec]:
    return [s for s in MODEL_SPECS if results.load_metrics(s.metrics_file) is None]


def best_model(table: pd.DataFrame) -> pd.Series | None:
    models = table[table["Type"] != "Reference"]
    if models.empty:
        return None
    return models.loc[models["F1 (macro)"].idxmax()]


@st.cache_data(show_spinner=False)
def metric_long_format() -> pd.DataFrame:
    """Long format for the grouped bar chart: Model, Metric, Score."""
    table = comparison_table()
    metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    if table.empty:
        return pd.DataFrame(columns=["Model", "Metric", "Score"])
    return table.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")


@st.cache_data(show_spinner=False)
def per_class_f1() -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        m = results.load_metrics(spec.metrics_file)
        if not m:
            continue
        for label in LABELS:
            pc = m["per_class"][label]
            rows.append({"Model": spec.name, "Sentiment": label, "F1": pc["f1"],
                         "Precision": pc["precision"], "Recall": pc["recall"], "Support": pc["support"]})
    return pd.DataFrame(rows)


def model_colors() -> dict[str, str]:
    colors = {spec.name: spec.color for spec in MODEL_SPECS}
    colors[BASELINE_NAME] = BASELINE_COLOR
    return colors


def confusion_images() -> list[tuple[ModelSpec, object]]:
    return [(spec, results.result_image(spec.confusion_png)) for spec in MODEL_SPECS]


@st.cache_data(show_spinner=False)
def distilbert_loss_curves() -> pd.DataFrame | None:
    log = results.load_distilbert_training_log()
    if log is None:
        return None
    return log.melt(id_vars=["step", "epoch"], value_vars=["train_loss", "val_loss"],
                    var_name="Curve", value_name="Loss").replace(
        {"Curve": {"train_loss": "Training loss", "val_loss": "Validation loss"}})


@st.cache_data(show_spinner=False)
def distilbert_val_f1() -> pd.DataFrame | None:
    log = results.load_distilbert_training_log()
    return None if log is None else log[["step", "epoch", "val_macro_f1"]]


def distilbert_metrics() -> dict | None:
    return results.load_metrics(next(s.metrics_file for s in MODEL_SPECS if s.key == "distilbert"))


@st.cache_data(show_spinner=False)
def top_words() -> dict | None:
    """Most positive-/negative-weighted TF-IDF words from the Logistic Regression model."""
    m = results.load_metrics("logistic_regression_test_metrics.json")
    return m.get("top_features") if m else None


@st.cache_data(show_spinner=False)
def dataset_summary() -> dict | None:
    return results.load_dataset_summary()
