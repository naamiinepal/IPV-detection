import json
import os.path
from ast import literal_eval
from typing import Iterable, List, Mapping, Set, Tuple
from unicodedata import normalize

import pandas as pd

DATA_DIR = os.path.join("datasets", "word")


def convert_list_to_id_maker(label2id: Mapping[str, id]):
    def convert_list_to_id(labels: Iterable[str]):
        return tuple(label2id[lab] for lab in labels)

    return convert_list_to_id


def strip_and_normalize_nfkc(text: str):
    return normalize("NFKC", text.strip())


def tokens_cleaner(token_list: Iterable[str]):
    return tuple(map(strip_and_normalize_nfkc, token_list))


def label_names_sort_key(cat: str):
    # Sort by the later part first, then by first part
    return cat.split("-", 1)[::-1]


def convert_ac_to_ids(ac_series: pd.Series) -> Tuple[pd.Series, List[str]]:
    """Convert the aspects from list of strings to a list of integers

    Args:
        ac_series (pd.Series): The series containing the list of strings

    Returns:
        Tuple[pd.Series, List[str]]: The series with the list of integers,
            and the label_names
    """

    # Get all the tokens that are in the dataset
    total_labels: Set[str] = set()
    for ac in ac_series.apply(set):
        total_labels |= ac
    total_labels.discard("O")

    label_names = list(total_labels)
    label_names.sort(key=label_names_sort_key)

    # label_names when printed
    # ['B-Others',
    #  'I-Others',
    #  'B-character_assasination',
    #  'I-character_assasination',
    #  'B-ethnic_violence',
    #  'I-ethnic_violence',
    #  'B-general_threat',
    #  'I-general_threat',
    #  'B-physical_threat',
    #  'I-physical_threat',
    #  'B-profanity',
    #  'I-profanity',
    #  'B-rape_threat',
    #  'I-rape_threat',
    #  'B-religion_violence',
    #  'I-religion_violence',
    #  'B-sexism',
    #  'I-sexism']

    # Insert no entity token at the top
    label_names.insert(0, "O")

    print(label_names)

    label2id = {l: i for i, l in enumerate(label_names)}

    convert_list_to_id = convert_list_to_id_maker(label2id)

    return ac_series.apply(convert_list_to_id), label_names


def processed_word_df(save_df: bool, save_label_names: bool) -> pd.DataFrame:
    """
    Returns a df with tokens and aspect categorizes.
    Each cell in the dataframe is a tuple.
    The tokens are stripped and normalized to NFKC.
    Aspects categories are returned as integers,
        where the actual strings is saved in label_names.

    Args:
        save_df (bool: Whether to save the dataframe
        save_label_names (bool): Whether to save the label names as JSON

    Returns:
        pd.DataFrame: Dataframe with tokens normalized and aspects in integer
    """

    df: pd.DataFrame = (
        pd.read_csv(os.path.join(DATA_DIR, "overall.csv"), usecols=("tokens", "ac"))
        .applymap(literal_eval)
        .applymap(tuple)
    )

    # Normalize texts
    df["tokens"] = df["tokens"].apply(tokens_cleaner)

    print("Before")
    print(df.info())
    initial_count = len(df)

    # Remove rows with same tokens and same annotations
    df.drop_duplicates(inplace=True)

    print("\n\nAfter")
    print(df.info())

    print("\n\nDuplicates", initial_count - len(df))

    integer_ac, label_names = convert_ac_to_ids(df["ac"])

    if save_label_names:
        with open(os.path.join(DATA_DIR, "label_names.json"), "w") as f:
            json.dump(label_names, f)

    df["ac"] = integer_ac

    if save_df:
        df.to_csv(os.path.join(DATA_DIR, "combined.csv"), index=False)

    return df


if __name__ == "__main__":
    # Save both the dataframe and the label names
    processed_word_df(save_df=True, save_label_names=True)
