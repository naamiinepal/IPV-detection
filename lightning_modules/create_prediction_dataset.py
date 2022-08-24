import os.path
import re
from glob import glob
from unicodedata import normalize

import pandas as pd

url_pattern = re.compile(r"(https?://[^\s]+)")
mention_pattern = re.compile(r"@[^\s]+")
pattern1 = re.compile(
    r"(_[a-zA-Z0-9]+)|(#[\u0900-\u097F]+)|(@[\u0900-\u097F]+)|(_[\u0900-\u097F]+)"
)
pattern2 = re.compile(r"(\W)(?=\1)")
multi_whitespaces = re.compile(r"\s+")

to_replace = """@#=/+…:"")(}{][*%_’‘'"""


def cleaner(text: str) -> str:
    """
    Cleans the tweets using the fllowing sequential steps:
        - Remove all the words starting with '_' ("_ahiraj"),
            mentions starting with '_' ("@_Silent__Eyes__") and also
            Devanagiri ("@पाेखरा") and hashtags used with Devanagiri ("#पाेखरा").
        - Remove punctuations (selected manually).
            This does not include sentence enders like "|" or "." or "?" or "!".
        - Removes bad characters like "&gt;".
        - If a punctuation or a whitespace has been repeated multiple times,
            adjust it to a single occurence.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    no_url = re.sub(url_pattern, "", text)
    no_mention = re.sub(mention_pattern, "", no_url)
    new_text = (
        re.sub(pattern1, "", no_mention)
        .translate(str.maketrans("", "", to_replace))
        .translate(str.maketrans("", "", "&gt;"))
    )
    remove_repitition = re.sub(multi_whitespaces, " ", re.sub(pattern2, "", new_text))
    final_text = normalize("NFKC", remove_repitition).strip()
    return final_text


DATA_DIR = os.path.join("datasets", "raw")

file_names = glob(os.path.join(DATA_DIR, "*_tweets", "*.csv"))

dataframe_sequence = []
for file_name in file_names:
    try:
        df = pd.read_csv(file_name, usecols=["text"])
    except pd.errors.EmptyDataError:
        print("Empty dataframe:", file_name)
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
