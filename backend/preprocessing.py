"""Text preprocessing shared by the app and the export script.

These functions mirror the team's cleaning in notebooks/Reviews_Feature_Engineering.ipynb
(also copied into Logistic_Regression.ipynb and DistilBERT_Training.ipynb), so a review typed
into the app is transformed exactly like the reviews the models were trained on.
"""

import re

import pandas as pd

from backend.config import SENTIMENT_MAP

_HTML_TAG = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Remove HTML tags (e.g. <br />) and collapse whitespace."""
    text = _HTML_TAG.sub(" ", str(text))
    return " ".join(text.split())


def combine_review(summary: str | None, text: str) -> str:
    """Build the model input: cleaned summary + " " + cleaned review text.

    The models were trained on "CombinedText" = CleanSummary + " " + CleanText.
    When the user gives no summary/title we just use the review text.
    """
    summary = clean_text(summary) if summary else ""
    body = clean_text(text)
    return f"{summary} {body}".strip()


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the team's full cleaning pipeline to the raw 50k sample.

    1) drop rows with a missing Summary   2) drop duplicate review Text (keep first)
    3) map stars to Negative / Neutral / Positive   4) build CombinedText
    """
    clean_df = df.dropna(subset=["Summary"]).copy()
    clean_df = clean_df.drop_duplicates(subset=["Text"]).copy()
    clean_df["Sentiment"] = clean_df["Score"].map(SENTIMENT_MAP)
    clean_df["CleanSummary"] = clean_df["Summary"].apply(clean_text)
    clean_df["CleanText"] = clean_df["Text"].apply(clean_text)
    clean_df["CombinedText"] = clean_df["CleanSummary"] + " " + clean_df["CleanText"]
    return clean_df
