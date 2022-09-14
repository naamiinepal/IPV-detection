import os.path
from glob import glob

import numpy as np
import pandas as pd

DATA_DIR = os.path.join("datasets", "sentence")

file_names = glob(os.path.join(DATA_DIR, "*.csv"))


dtype = {"text": object, "abuse": np.uint8, "sexual_content_score": np.uint8}

dataframe_sequence = [
    pd.read_csv(file_name, usecols=dtype.keys()) for file_name in file_names
]

combined_df = pd.concat(dataframe_sequence)

print("Before")
print(combined_df.info())
initial_count = len(combined_df)

#  Normalize and strip spaces
combined_df["text"] = combined_df["text"].str.normalize("NFKC").str.strip()

# Drop for same annotator and same annotation
combined_df = combined_df.fillna(0).drop_duplicates().convert_dtypes()

print("\n\nAfter")
print(combined_df.info())

print("\n\nDuplicates", initial_count - len(combined_df))

combined_df.to_csv(os.path.join(DATA_DIR, "combined.csv"), index=False)
