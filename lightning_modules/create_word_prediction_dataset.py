#! .venv/bin/python

import os.path
from glob import glob

import pandas as pd

from utils import cleaner

DATA_DIR = os.path.join("datasets", "raw_words")

file_names = glob(os.path.join(DATA_DIR, "*_tweets", "*.csv"))

dataframe_sequence = []
for file_name in file_names:
    try:
        df = pd.read_csv(file_name, usecols=["text"])
    except Exception:
        print("Skipping", file_name)
    else:
        dataframe_sequence.append(df)

combined_df = pd.concat(dataframe_sequence).dropna()

print("Before")
print(combined_df.info())
initial_count = len(combined_df)

#  Normalize and strip spaces
preprocessed_text = combined_df["text"].apply(cleaner).drop_duplicates()

condition = preprocessed_text.apply(lambda x: len(x.split()) >= 3)

old_texts = preprocessed_text[condition]

training_texts = pd.read_csv(
    os.path.join("datasets", "sentence", "combined.csv"), usecols=["text"]
)["text"]

new_text = set(old_texts) - set(training_texts)
combined_df = pd.DataFrame({"text": tuple(new_text)})

print("\n\nAfter")
print(combined_df.info())

print("\n\nDuplicates", initial_count - len(combined_df))

combined_df.to_csv(os.path.join(DATA_DIR, "combined.csv"), index=False)
