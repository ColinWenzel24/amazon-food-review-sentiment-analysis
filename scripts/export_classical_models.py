"""Export the classical model(s) so the Streamlit app can load them.

Karan's Logistic Regression (notebooks/Logistic_Regression.ipynb) lives only inside the notebook.
This script reproduces it EXACTLY - same cleaning, same 80/20 stratified split (seed 48),
TfidfVectorizer() with default settings, LogisticRegression(max_iter=1000) - and saves it.
The notebook itself is not modified.

Outputs
  models/logistic_regression_tfidf.joblib           fitted Pipeline(tfidf -> logistic regression)
  results/logistic_regression_test_metrics.json     same schema as distilbert_test_metrics.json
  results/logistic_regression_confusion_matrix.png
  results/majority_baseline_test_metrics.json
  results/dataset_summary.json                      numbers for the app's Dataset tab

Usage (from the repo root, after `python data/make_sample.py`):
  python scripts/export_classical_models.py
  python scripts/export_classical_models.py --data path/to/group_sample_seed48.csv
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from backend.config import (BASELINE_METRICS_FILE, DATASET_SUMMARY_FILE, LABELS,  # noqa: E402
                            MODELS_DIR, RESULTS_DIR, SAMPLE_CSV, SEED)
from backend.preprocessing import prepare_dataframe  # noqa: E402

# Value printed by Logistic_Regression.ipynb ("Misclassified reviews: 1224 of 9253").
# If this script matches it, the exported model is the same model as Karan's.
NOTEBOOK_MISCLASSIFIED = 1224


def metrics_dict(name, y_true, y_pred, training_seconds, majority, extra=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    per = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    out = {
        "model": name,
        "test_size": int(len(y_true)),
        "accuracy": acc,
        "macro_precision": p, "macro_recall": r, "macro_f1": f1,
        "weighted_precision": wp, "weighted_recall": wr, "weighted_f1": wf1,
        "majority_baseline_accuracy": majority,
        "training_minutes": training_seconds / 60,
        "training_seconds": training_seconds,
        "device": "cpu",
        "per_class": {
            LABELS[i]: {"precision": per[0][i], "recall": per[1][i], "f1": per[2][i], "support": int(per[3][i])}
            for i in range(len(LABELS))
        },
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
    if extra:
        out.update(extra)
    return out


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    print("  wrote", path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(SAMPLE_CSV), help="path to group_sample_seed48.csv")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}\nRun `python data/make_sample.py` first (see data/README.md).")

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- 1. Load + clean exactly like the notebooks ----
    raw_df = pd.read_csv(data_path)
    clean_df = prepare_dataframe(raw_df)
    print(f"Rows: {len(raw_df):,} raw -> {len(clean_df):,} after cleaning (expected 46,261)")

    train_df, test_df = train_test_split(
        clean_df, test_size=0.20, random_state=SEED, stratify=clean_df["Sentiment"]
    )
    X_train, X_test = train_df["CombinedText"], test_df["CombinedText"]
    y_train, y_test = train_df["Sentiment"], test_df["Sentiment"]
    print(f"Train: {len(train_df):,}   Test: {len(test_df):,}")

    # ---- 2. Majority baseline ----
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    majority = float(accuracy_score(y_test, base_pred))
    save_json(metrics_dict("majority_baseline", y_test, base_pred, 0.0, majority),
              RESULTS_DIR / BASELINE_METRICS_FILE)

    # ---- 3. Logistic Regression on TF-IDF (Karan's settings) ----
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    y_pred = pipeline.predict(X_test)

    n_wrong = int((y_pred != y_test.to_numpy()).sum())
    status = "MATCHES" if n_wrong == NOTEBOOK_MISCLASSIFIED else "DIFFERS FROM"
    print(f"Logistic Regression: {n_wrong} misclassified -> {status} the notebook ({NOTEBOOK_MISCLASSIFIED})")

    vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
    clf = pipeline.named_steps["clf"]
    top_features = {}
    for label in LABELS:
        weights = pd.Series(clf.coef_[list(clf.classes_).index(label)], index=vocab)
        top_features[label] = [[w, round(float(v), 4)] for w, v in weights.nlargest(15).items()]

    lr_metrics = metrics_dict(
        "logistic_regression_tfidf", y_test, y_pred, train_seconds, majority,
        extra={
            "vocabulary_size": int(len(vocab)),
            "top_features": top_features,
            "settings": {"vectorizer": "TfidfVectorizer()", "classifier": "LogisticRegression(max_iter=1000)",
                         "seed": SEED, "test_size": 0.20},
            "matches_notebook": n_wrong == NOTEBOOK_MISCLASSIFIED,
        },
    )
    save_json(lr_metrics, RESULTS_DIR / "logistic_regression_test_metrics.json")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=LABELS), display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Logistic Regression confusion matrix (test set)")
    plt.tight_layout()
    png = RESULTS_DIR / "logistic_regression_confusion_matrix.png"
    plt.savefig(png, dpi=200)
    print("  wrote", png.relative_to(ROOT))

    model_path = MODELS_DIR / "logistic_regression_tfidf.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "metadata": {"sklearn_version": sklearn.__version__, "train_size": int(len(train_df)),
                     "seed": SEED, "source_notebook": "notebooks/Logistic_Regression.ipynb"},
    }, model_path, compress=3)
    print(f"  wrote {model_path.relative_to(ROOT)} ({model_path.stat().st_size / 1e6:.1f} MB)")

    # ---- 4. Dataset summary for the app ----
    words = clean_df["CleanText"].str.split().str.len()
    bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, np.inf]
    bin_labels = ["0–25", "26–50", "51–75", "76–100", "101–150", "151–200", "201–300", "301–500", "500+"]
    length_hist = pd.cut(words, bins=bins, labels=bin_labels, right=True).value_counts().reindex(bin_labels)
    by_class_len = clean_df.assign(words=words).groupby("Sentiment")["words"].median()

    summary = {
        "raw_rows": int(len(raw_df)),
        "missing_summary": int(raw_df["Summary"].isna().sum()),
        "duplicate_texts_removed": int(raw_df.dropna(subset=["Summary"]).duplicated(subset=["Text"]).sum()),
        "html_reviews": int(raw_df["Text"].str.contains(r"<[^>]+>").sum()),
        "clean_rows": int(len(clean_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "score_counts": {str(k): int(v) for k, v in clean_df["Score"].value_counts().sort_index().items()},
        "class_counts": {k: int(clean_df["Sentiment"].value_counts()[k]) for k in LABELS},
        "avg_words": float(words.mean()),
        "median_words": float(words.median()),
        "median_words_by_class": {k: float(by_class_len[k]) for k in LABELS},
        "length_histogram": {k: int(v) for k, v in length_hist.items()},
        "vocabulary_size": int(len(vocab)),
    }
    save_json(summary, RESULTS_DIR / DATASET_SUMMARY_FILE)
    print("Done.")


if __name__ == "__main__":
    main()
