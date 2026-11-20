# Saved Models

This folder will contain trained model files used for
evaluation and the Streamlit application.

## DistilBERT model (deep model)

The trained DistilBERT model is too big for GitHub (268 MB), so it is hosted on Hugging Face:
https://huggingface.co/thekluna/amazon-food-sentiment-distilbert

Load it in Python:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("thekluna/amazon-food-sentiment-distilbert")
model = AutoModelForSequenceClassification.from_pretrained("thekluna/amazon-food-sentiment-distilbert")
```

Labels: 0 = Negative, 1 = Neutral, 2 = Positive.
Training code: notebooks/DistilBERT_Training.ipynb
