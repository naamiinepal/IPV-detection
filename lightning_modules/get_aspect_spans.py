#! .venv/bin/python

from typing import List, Tuple
import pandas as pd
import os.path
from string import punctuation

from utils import tokenizer_normalize

DATA_DIR = os.path.join("datasets", "raw_words")

df = pd.read_csv(os.path.join(DATA_DIR, "predictions.csv"))

muril_normalize = tokenizer_normalize()
df["text"] = df["text"].apply(muril_normalize)

overall_spans = []

punc_texts = set()

for row in df.itertuples(index=False):
    # Convert strings to lists
    splitted_texts = eval(row.splitted_texts)
    aspects = eval(row.aspects)
    text = row.text

    spans: List[Tuple[int, int, str]] = []
    start = 0

    for (token, asp) in zip(splitted_texts, aspects):
        token = token.lstrip("##")
        index = text.find(token, start)
        if index == -1:
            print(text, token)
            continue
        start = index + len(token)
        if asp != "O":
            spans.append((index, start, asp))

    updated_spans = []
    i = 0
    while i < len(spans):
        start, end, prev_asp = spans[i]

        # Remove B- and I- prefixes
        prev_asp = prev_asp[2:]

        unbroken = True
        for j in range(i + 1, len(spans)):
            curr_start, curr_end, curr_asp = spans[j]

            # Remove B- and I- prefixes
            curr_asp = curr_asp[2:]

            curr_text = text[end:curr_start].strip()

            only_punc = all(i in punctuation for i in curr_text)

            if curr_text and only_punc:
                punc_texts.add(text)

            if prev_asp != curr_asp or not only_punc:
                unbroken = False
                break
            end = curr_end
        updated_spans.append((start, end, prev_asp))

        # If not broken, then skip the last iteration
        if unbroken:
            break
        i = j

    overall_spans.append(updated_spans)

df["aspect_anno"] = overall_spans

print("Punctuation separator", punc_texts)

df.to_csv(os.path.join(DATA_DIR, "span_predictions.csv"), index=False)
