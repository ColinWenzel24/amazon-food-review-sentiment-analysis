# Amazon Food Review Sentiment Analysis

## Project Description
This CSE 3521 group project uses machine learning to classify
Amazon food reviews as positive, neutral, or negative.
We will compare classical and deep learning models and build
a Streamlit application for predicting sentiment from new reviews.

**Live app:** _coming soon (Streamlit Community Cloud URL)_

## Team Members
- Colin Wenzel — Data Engineer
- KC Luna — ML Engineer
- Jackson Ball — Analyst
- Abhinav Siriguri — Team Lead
- Karan Tandra - ML Engineer
- Ardelia Putridaryana - App Engineer

## Dataset
We use a shared sample of 50,000 reviews from the
Amazon Fine Food Reviews dataset.

Dataset source:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## Results so far (held-out test set, 9,253 reviews)

| Model | Features | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|
| Majority baseline | — | 0.780 | 0.26 | 0.33 | 0.29 |
| Logistic Regression | TF-IDF | 0.868 | 0.71 | 0.59 | 0.62 |
| DistilBERT | fine-tuned transformer | 0.883 | 0.730 | 0.786 | 0.748 |

The full comparison (with training time, per-class scores and confusion matrices) is in the app's
**Model comparison** tab, generated from the JSON files in `results/`.

## Repository structure

```
app.py                     Streamlit entry point (streamlit run app.py)
backend/                   no Streamlit code: preprocessing, model loading, inference, results I/O
  config.py                paths, labels, list of served models (MODEL_SPECS)
  preprocessing.py         same cleaning as the notebooks (HTML removal, title + text)
  models.py                wrappers for the scikit-learn pipeline and DistilBERT
  results.py               reads metrics/curves saved in results/
controllers/               between UI and backend: input validation, model caching, table building
  prediction_controller.py single-review and batch predictions
  insights_controller.py   model comparison table, training curves, dataset summary
frontend/                  everything the user sees
  theme.py                 Amazon-inspired CSS
  components.py            navbar, prediction cards, alerts, stat tiles
  charts.py                Altair charts
  views/                   one module per tab (analyze, batch, compare, dataset, about)
  assets/sample_reviews.csv
scripts/
  export_classical_models.py  rebuilds and saves the Logistic Regression model for the app
data/                      instructions + make_sample.py (CSV not committed)
notebooks/                 EDA, feature engineering, Logistic Regression, DistilBERT training
models/                    saved classical models (DistilBERT is on Hugging Face)
results/                   test metrics, confusion matrices, training logs
report/                    report PDFs and slides
.streamlit/config.toml     app theme
```

## Reproducing our results

**1. Install dependencies** (Python 3.12 recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Get the data.** Download `Reviews.csv` from
[Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), then create the shared
50,000-review sample (seed 48). See [data/README.md](data/README.md):

```bash
python data/make_sample.py /path/to/Reviews.csv   # writes data/group_sample_seed48.csv
```

**3. Retrain the models**

- *EDA / features:* `notebooks/Amazon_Reviews_Analysis.ipynb`, `notebooks/Reviews_Feature_Engineering.ipynb`
- *Logistic Regression:* `notebooks/Logistic_Regression.ipynb` (analysis). To save the model for the app:
  ```bash
  python scripts/export_classical_models.py
  ```
  This uses the same cleaning, split and hyperparameters as the notebook (and checks that it gets the same
  1,224 test errors), then writes `models/logistic_regression_tfidf.joblib`, the test metrics and
  confusion matrix in `results/`, and `results/dataset_summary.json`.
- *DistilBERT:* `notebooks/DistilBERT_Training.ipynb` (GPU recommended, about 4 minutes on a modern GPU). The
  trained model is published at
  [thekluna/amazon-food-sentiment-distilbert](https://huggingface.co/thekluna/amazon-food-sentiment-distilbert).

**4. Run the app**

```bash
streamlit run app.py
```

The app downloads DistilBERT from Hugging Face on first start (about 270 MB, cached afterwards). To use a
local copy instead, put it in `models/distilbert_sentiment/` or set `DISTILBERT_MODEL=/path/to/model`.

## The Streamlit app

| Tab | What it does |
|---|---|
| Analyze a review | Type/paste a review (+ optional title) or pick an example; see Logistic Regression and DistilBERT predictions side by side with class probabilities, an agreement check, and the words that drove the Logistic Regression decision. |
| Batch analysis | Upload a CSV of reviews (up to 500), score them with each model, view the sentiment mix and disagreements, and download the results. |
| Model comparison | Unified table (accuracy, precision, recall, F1, training time) vs. the majority baseline, per-class F1, confusion matrices, DistilBERT training/validation loss curves. |
| The data | Dataset size, star and label distributions, review lengths, cleaning steps, most informative words. |
| How it works | Plain-language pipeline explanation, model descriptions, team, technical status. |

### Deploying to Streamlit Community Cloud
1. Make sure `models/logistic_regression_tfidf.joblib` and the `results/` files are committed.
2. At [share.streamlit.io](https://share.streamlit.io), click **Create app**, choose this repository,
   branch `main`, main file `app.py`. Under **Advanced settings**, choose **Python 3.12**.
3. Deploy and paste the URL at the top of this README.

### Adding another model to the app
Save a fitted scikit-learn `Pipeline` (text → vectorizer → classifier) with `joblib` in `models/`, write its
test metrics to `results/<key>_test_metrics.json` (same keys as `distilbert_test_metrics.json`), and add a
`ModelSpec` entry in `backend/config.py`. The app picks it up in every tab automatically.
