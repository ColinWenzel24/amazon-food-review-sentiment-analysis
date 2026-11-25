"""Central configuration: paths, labels, and the list of models the app serves."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

SAMPLE_CSV = DATA_DIR / "group_sample_seed48.csv"
SEED = 48  # Group Psi seed: same as data/make_sample.py and every notebook

# Label scheme used by every model (see notebooks/Amazon_Reviews_Analysis.ipynb)
LABELS = ["Negative", "Neutral", "Positive"]
SENTIMENT_MAP = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
STAR_RANGE = {"Negative": "1–2 ★", "Neutral": "3 ★", "Positive": "4–5 ★"}

# DistilBERT: use a local copy if a teammate has one, otherwise download from Hugging Face.
# Override with the DISTILBERT_MODEL environment variable.
DISTILBERT_LOCAL_DIR = MODELS_DIR / "distilbert_sentiment"
DISTILBERT_HF_REPO = "thekluna/amazon-food-sentiment-distilbert"
DISTILBERT_MAX_LENGTH = 256  # same as training (notebooks/DistilBERT_Training.ipynb)


def distilbert_source() -> str:
    env = os.environ.get("DISTILBERT_MODEL")
    if env:
        return env
    if (DISTILBERT_LOCAL_DIR / "config.json").exists():
        return str(DISTILBERT_LOCAL_DIR)
    return DISTILBERT_HF_REPO


@dataclass(frozen=True)
class ModelSpec:
    """One servable model.

    key          short id used in code and in results file names
    name         display name
    family       "classical" or "deep"
    features     the text representation the model uses
    loader       "sklearn" (a joblib-saved Pipeline) or "hf" (Hugging Face transformer)
    artifact     path to the saved model (sklearn) - HF models resolve via distilbert_source()
    metrics_file JSON produced by the training/export code, in results/
    owner        teammate who built the model
    """

    key: str
    name: str
    family: str
    features: str
    loader: str
    metrics_file: str
    owner: str
    artifact: Path | None = None
    confusion_png: str | None = None
    color: str = "#007EA8"


# To serve another classical model (e.g. Naive Bayes), save it as a fitted sklearn Pipeline
# (text -> features -> classifier) and add one ModelSpec here. Nothing else needs to change.
MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        key="logistic_regression",
        name="Logistic Regression",
        family="classical",
        features="TF-IDF (unigrams, 38k-word vocabulary)",
        loader="sklearn",
        artifact=MODELS_DIR / "logistic_regression_tfidf.joblib",
        metrics_file="logistic_regression_test_metrics.json",
        confusion_png="logistic_regression_confusion_matrix.png",
        owner="Karan Tandra",
        color="#007EA8",
    ),
    ModelSpec(
        key="distilbert",
        name="DistilBERT",
        family="deep",
        features="WordPiece tokens → contextual embeddings (fine-tuned transformer)",
        loader="hf",
        metrics_file="distilbert_test_metrics.json",
        confusion_png="confusion_matrix.png",
        owner="KC Luna",
        color="#C45500",
    ),
]

BASELINE_METRICS_FILE = "majority_baseline_test_metrics.json"
DATASET_SUMMARY_FILE = "dataset_summary.json"
DISTILBERT_TRAINING_LOG = "distilbert_training_log.csv"
DISTILBERT_LOSS_PNG = "loss_curves.png"

# Limits that keep the deployed app responsive on Streamlit Community Cloud (CPU only)
MAX_REVIEW_CHARS = 5000
MAX_BATCH_ROWS = 500
