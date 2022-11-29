#! .venv/bin/python

from string import punctuation
from typing import Callable, List, Tuple
from ast import literal_eval

import pandas as pd


def get_aspect_spans(
    df: pd.DataFrame,
    splitted_texts_colname: str,
    aspects_colname: str,
    normalizer: Callable[[str], str],
) -> List[Tuple[int, int, str]]:
    """Get aspect spans from a dataframe."""

    overall_spans = []

    punc_texts = set()

    df["text"] = df["text"].apply(normalizer)
    df[[splitted_texts_colname, aspects_colname]] = df[
        [splitted_texts_colname, aspects_colname]
    ].applymap(literal_eval)

    for row in df.itertuples(index=False):
        # Convert strings to lists
        splitted_texts = map(normalizer, getattr(row, splitted_texts_colname))
        aspects = getattr(row, aspects_colname)
        text = row.text

        spans: List[Tuple[int, int, str]] = []
        start = 0

        for (token, asp) in zip(splitted_texts, aspects):
            token = token.lstrip("##")
            index = text.find(token, start)
            if index == -1:
                print("\n\nText:", text)
                print("Token:", token, end="\n\n")
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
                    punc_texts.add(curr_text)

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

    print("\nPunctuation separator", punc_texts, end="\n\n")

    return overall_spans


if __name__ == "__main__":
    import os.path
    from argparse import ArgumentParser, BooleanOptionalAction

    from utils import tokenizer_normalize

    parser = ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the CSV file",
        default=os.path.join("datasets", "raw_words", "predictions.csv"),
    )

    parser.add_argument(
        "--splitted_texts_colname",
        type=str,
        help="Name of the column with splitted texts",
        default="splitted_texts",
    )
    parser.add_argument(
        "--aspects_colname",
        type=str,
        help="Name of the column with aspects predictions",
        default="aspects",
    )

    parser.add_argument(
        "--include_splitted_aspects",
        default=True,
        help=(
            "Whether to include splitted texts and "
            "aspects columns in the output CSV file"
        ),
        action=BooleanOptionalAction,
    )

    args = parser.parse_args()

    predictions_df = pd.read_csv(args.csv_path)

    muril_normalize = tokenizer_normalize()

    overall_spans = get_aspect_spans(
        predictions_df,
        args.splitted_texts_colname,
        args.aspects_colname,
        muril_normalize,
    )

    predictions_df["aspect_anno"] = overall_spans

    if not args.include_splitted_aspects:
        print("\nRemoving splitted texts and aspects columns\n")
        predictions_df = predictions_df.drop(
            columns=[args.splitted_texts_colname, args.aspects_colname]
        )

    base_dir, filename = os.path.split(args.csv_path)

    predictions_df.to_csv(os.path.join(base_dir, f"span_{filename}"), index=False)
