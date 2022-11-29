#! .venv/bin/python

import os.path

import pandas as pd

from utils import tokenizer_normalize


def expand_annotator(annotator: str):
    anno = annotator.split("_", maxsplit=1)[0]
    if anno == "krn":
        return "kiran"
    if anno == "shr":
        return "sharmila"
    print("\nNew Annotator: ", anno, end="\n\n")
    return anno


def link_to_username(link: str):
    if isinstance(link, str):
        splitted = link.rsplit("/", 3)
        if len(splitted) == 4:
            return splitted[1]
    return link


DATA_DIR = "datasets"

sentence_df = pd.read_csv(os.path.join(DATA_DIR, "sentence", "combined_with_dupes.csv"))

# Normalize the texts
muril_normalize = tokenizer_normalize()
sentence_df["text"] = sentence_df["text"].apply(muril_normalize)

# Already normalized when saved
word_span_df = pd.read_csv(
    os.path.join(DATA_DIR, "word", "span_overall.csv"),
    usecols=["annotator", "text", "aspect_anno", "date", "link"],
)


word_span_df["annotator"] = word_span_df["annotator"].apply(expand_annotator)

word_span_df["username"] = word_span_df["link"].apply(link_to_username)

word_span_df.drop(columns=["link"], inplace=True)

combined_df = pd.merge(sentence_df, word_span_df, on=["annotator", "text"], how="outer")

combined_df.dropna(subset=["abuse"], inplace=True)

combined_df.rename(
    columns={
        "date": "created_at",
        "abuse": "is_abuse",
        "sexual_content_score": "sexual_score",
        "aspect_anno": "aspects_anno",
    },
    inplace=True,
)

combined_df.to_csv("labeled_sent_word_combined.csv", index=False)
