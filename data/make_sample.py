import pandas as pd
import os
import sys

seed = 48 # Our group, don't change

default_path = os.path.join(os.path.expanduser('~'), "Downloads", "Reviews.csv")
reviews_path = sys.argv[1] if len(sys.argv) > 1 else default_path

df = pd.read_csv(reviews_path)
sample = df.sample(n=50000, random_state=seed)

# Sample csv goes to data folder
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"group_sample_seed{seed}.csv")
sample.to_csv(out_path, index=False)

print(f"Saved {len(sample)} rows to {out_path}")
print(sample[['Score', 'Text']].head(3))