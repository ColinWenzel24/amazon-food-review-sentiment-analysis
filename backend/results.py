"""Read the evaluation results the team saved in results/ (read-only)."""

import json
from pathlib import Path

import pandas as pd

from backend.config import (BASELINE_METRICS_FILE, DATASET_SUMMARY_FILE, DISTILBERT_TRAINING_LOG,
                            RESULTS_DIR)


def _read_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metrics(metrics_file: str) -> dict | None:
    return _read_json(metrics_file)


def load_baseline_metrics() -> dict | None:
    return _read_json(BASELINE_METRICS_FILE)


def load_dataset_summary() -> dict | None:
    return _read_json(DATASET_SUMMARY_FILE)


def load_distilbert_training_log() -> pd.DataFrame | None:
    path = RESULTS_DIR / DISTILBERT_TRAINING_LOG
    return pd.read_csv(path) if path.exists() else None


def result_image(name: str | None) -> Path | None:
    if not name:
        return None
    path = RESULTS_DIR / name
    return path if path.exists() else None
