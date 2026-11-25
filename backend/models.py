"""Model wrappers. Every model exposes the same interface:

    model.predict(texts: list[str]) -> list[Prediction]

Inputs are already-combined review strings (see preprocessing.combine_review).
These classes only *load and run* the teammates' trained models; they never train or change them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend.config import DISTILBERT_MAX_LENGTH, LABELS, ModelSpec, distilbert_source


@dataclass
class Prediction:
    label: str
    probabilities: dict[str, float]          # {"Negative": p, "Neutral": p, "Positive": p}
    latency_ms: float = 0.0
    # Words that pushed toward / away from the predicted label (classical models only)
    supporting_words: list[tuple[str, float]] = field(default_factory=list)
    opposing_words: list[tuple[str, float]] = field(default_factory=list)
    truncated: bool = False                  # DistilBERT only reads the first 256 tokens

    @property
    def confidence(self) -> float:
        return self.probabilities[self.label]


class ModelUnavailableError(RuntimeError):
    """Raised when a model's saved files are missing or cannot be loaded."""


class SklearnPipelineModel:
    """A fitted scikit-learn Pipeline (raw text -> vectorizer -> classifier) saved with joblib."""

    def __init__(self, spec: ModelSpec):
        import joblib

        self.spec = spec
        if spec.artifact is None or not spec.artifact.exists():
            raise ModelUnavailableError(
                f"{spec.name} file not found at models/{spec.artifact.name if spec.artifact else '?'}. "
                "Run `python scripts/export_classical_models.py` to create it."
            )
        bundle = joblib.load(spec.artifact)
        # The export script saves {"pipeline": Pipeline, "metadata": {...}}; accept a bare Pipeline too
        self.pipeline = bundle["pipeline"] if isinstance(bundle, dict) else bundle
        self.metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
        self.classes = list(self.pipeline.classes_)

        self._vectorizer = self.pipeline.steps[0][1]
        self._classifier = self.pipeline.steps[-1][1]
        self._can_explain = hasattr(self._classifier, "coef_") and hasattr(
            self._vectorizer, "get_feature_names_out"
        )
        if self._can_explain:
            self._vocab = self._vectorizer.get_feature_names_out()
            coef = np.asarray(self._classifier.coef_)
            # Weight of each word for a class *relative to the other classes*
            self._relative_coef = coef - coef.mean(axis=0, keepdims=True)

    def predict(self, texts: list[str], explain: bool = True, top_k: int = 6) -> list[Prediction]:
        start = time.perf_counter()
        probs = self.pipeline.predict_proba(texts)
        features = self._vectorizer.transform(texts) if (explain and self._can_explain) else None
        per_item_ms = (time.perf_counter() - start) * 1000 / max(1, len(texts))

        results = []
        for i, row in enumerate(probs):
            prob_map = {label: float(row[self.classes.index(label)]) for label in LABELS}
            label = max(prob_map, key=prob_map.get)
            pred = Prediction(label=label, probabilities=prob_map, latency_ms=per_item_ms)
            if features is not None:
                pred.supporting_words, pred.opposing_words = self._explain(features[i], label, top_k)
            results.append(pred)
        return results

    def _explain(self, x_row, label: str, top_k: int):
        """Contribution of each word in the review = TF-IDF value x class weight."""
        x_row = x_row.tocsr()
        if x_row.nnz == 0:
            return [], []
        class_idx = self.classes.index(label)
        contrib = x_row.data * self._relative_coef[class_idx, x_row.indices]
        order = np.argsort(contrib)
        words = self._vocab[x_row.indices]
        supporting = [(words[j], float(contrib[j])) for j in order[::-1][:top_k] if contrib[j] > 0]
        opposing = [(words[j], float(contrib[j])) for j in order[:top_k] if contrib[j] < 0]
        return supporting, opposing


class DistilBertModel:
    """KC's fine-tuned DistilBERT, loaded from Hugging Face (or a local copy)."""

    def __init__(self, spec: ModelSpec, batch_size: int = 16):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelUnavailableError(f"PyTorch/transformers not installed: {exc}") from exc

        self.spec = spec
        self.source = distilbert_source()
        self.batch_size = batch_size
        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.source)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.source)
        except Exception as exc:
            raise ModelUnavailableError(f"Could not load DistilBERT from '{self.source}': {exc}") from exc
        self.model.eval()
        # Map output index -> our label names using the model's own config (0=Neg, 1=Neu, 2=Pos)
        id2label = self.model.config.id2label
        self.index_labels = [id2label.get(i, LABELS[i]) for i in range(len(id2label))]

    def predict(self, texts: list[str], **_) -> list[Prediction]:
        torch = self._torch
        results: list[Prediction] = []
        start = time.perf_counter()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            full_lengths = [
                len(ids) for ids in self.tokenizer(batch, truncation=False, verbose=False)["input_ids"]
            ]
            enc = self.tokenizer(
                batch, truncation=True, max_length=DISTILBERT_MAX_LENGTH, padding=True, return_tensors="pt"
            )
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            for row, n_tokens in zip(probs, full_lengths):
                prob_map = {label: 0.0 for label in LABELS}
                for idx, p in enumerate(row):
                    prob_map[self.index_labels[idx]] = float(p)
                label = max(prob_map, key=prob_map.get)
                results.append(
                    Prediction(label=label, probabilities=prob_map,
                               truncated=n_tokens > DISTILBERT_MAX_LENGTH)
                )
        per_item_ms = (time.perf_counter() - start) * 1000 / max(1, len(texts))
        for r in results:
            r.latency_ms = per_item_ms
        return results


def load_model(spec: ModelSpec):
    if spec.loader == "sklearn":
        return SklearnPipelineModel(spec)
    if spec.loader == "hf":
        return DistilBertModel(spec)
    raise ValueError(f"Unknown loader '{spec.loader}' for model '{spec.key}'")
