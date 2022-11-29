#! .venv/bin/python

import os.path
from glob import glob

import pandas as pd

from utils import cleaner, tokenizer_normalize

DATA_DIR = os.path.join("datasets", "raw_words")

file_names = glob(os.path.join(DATA_DIR, "*_tweets", "*.csv"))

dataframe_sequence = []
for file_name in file_names:
    try:
        df = pd.read_csv(file_name, usecols=["text", "user_name", "created_at"])
    except Exception:
        print("Skipping", file_name)
    else:
        dataframe_sequence.append(df)

combined_df = pd.concat(dataframe_sequence).dropna()

print("Before")
print(combined_df.info())
initial_count = len(combined_df)

#  Normalize and strip spaces
combined_df["text"] = combined_df["text"].apply(cleaner)

keep_condition = combined_df["text"].apply(lambda x: len(x.split()) >= 3)

muril_normalize = tokenizer_normalize()

combined_df["text"] = combined_df["text"].apply(muril_normalize)

combined_df = combined_df[keep_condition]

combined_df.drop_duplicates(("text", "created_at"), inplace=True)


prediction_texts = pd.read_csv(
    os.path.join(DATA_DIR, "span_predictions.csv"), usecols=["text", "aspect_anno"]
)

combined_df = pd.merge(prediction_texts, combined_df, on="text")

print("\n\nAfter")
print(combined_df.info())

print("\nSmall Tweets", initial_count - len(combined_df))

combined_df.to_csv(os.path.join(DATA_DIR, "combined_with_info.csv"), index=False)