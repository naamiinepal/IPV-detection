#! .venv/bin/python

import os.path

import pandas as pd
from utils import tokenizer_normalize

ROOT_DIR = "datasets"

SENT_DIR = os.path.join(ROOT_DIR, "raw")
WORD_DIR = os.path.join(ROOT_DIR, "raw_words")

PREDICTIONS_FILE = "combined_with_info.csv"

sent_df = pd.read_csv(os.path.join(SENT_DIR, PREDICTIONS_FILE))

muril_tokenize_detokenize = tokenizer_normalize()

sent_df["text"] = sent_df["text"].apply(muril_tokenize_detokenize)

word_df = pd.read_csv(os.path.join(WORD_DIR, PREDICTIONS_FILE))

print("Merging dataframes...")

combined_df = pd.merge(sent_df, word_df, on=["text", "user_name", "created_at"])

print("Saving combined dataframe...")

combined_df.to_csv(os.path.join(ROOT_DIR, "word_sent_combined.csv"), index=False)
