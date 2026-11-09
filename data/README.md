# Dataset

This project uses a shared sample of 50,000 reviews from the
Amazon Fine Food Reviews dataset.

## Original Dataset

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## Shared Group Sample

Filename: group_sample_seed48.csv

How to obtain sample: 

Your group seed

Group	Seed
Group Psi	48


Step 1—Download the raw dataset (one time).

Go to https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews, sign in, and click Download. Unzip the archive—the file you need is Reviews.csv.

Step 2 — Run this script to generate your group sample

Replace YOUR_SEED with your group's number and PATH_TO_REVIEWS_CSV with where you saved the file.

import pandas as pd
import os

# --- EDIT THESE TWO LINES ---
seed = YOUR_SEED                          # e.g. 42 for Group Alpha
reviews_path = "PATH_TO_REVIEWS_CSV"     # e.g. "/Users/yourname/Downloads/Reviews.csv"
# ----------------------------

df = pd.read_csv(reviews_path)
sample = df.sample(n=50000, random_state=seed)

out_path = os.path.join(os.path.expanduser("~"), "Downloads", f"group_sample_seed{seed}.csv")
sample.to_csv(out_path, index=False)

print(f"Saved {len(sample)} rows to {out_path}")
print(sample[['Score', 'Text']].head(3))
The output file in your Downloads folder is your team's dataset for the entire project. Keep it—do not re-run with a different seed.
