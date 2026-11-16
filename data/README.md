# Dataset

This project uses a shared sample of 50,000 reviews from the
Amazon Fine Food Reviews dataset.

## Original Dataset

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## Shared Group Sample

Filename: `group_sample_seed48.csv` (Group Psi, seed 48)

**Step 1 — Download the raw dataset (one time).**
Go to https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews, sign in,
and click Download. Unzip the archive; the file you need is `Reviews.csv`.

**Step 2 — Generate the group sample.** From the repo root (requires pandas):

```bash
python3 data/make_sample.py                        # uses ~/Downloads/Reviews.csv
python3 data/make_sample.py /path/to/Reviews.csv   # or a custom path
```

This saves `data/group_sample_seed48.csv` (ignored by git). The first row
printed should be Id 515539; If it matches, your sample is identical to
everyone else's.
