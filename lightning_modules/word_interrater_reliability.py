import os.path
from ast import literal_eval
from typing import Literal

import krippendorff
import numpy as np
import pandas as pd

from preprocess_word import convert_ac_to_ids, tokens_cleaner

DATA_DIR = os.path.join("datasets", "word")
ANNOTATOR_COLUMN = "annotator"

JOIN_TYPE: Literal["inner", "outer"] = "outer"

df: pd.DataFrame = pd.read_csv(
    os.path.join(DATA_DIR, "overall.csv"), usecols=(ANNOTATOR_COLUMN, "tokens", "ac")
)


def get_annotator_df(annotator: str):
    new_df = (
        df[df[ANNOTATOR_COLUMN] == annotator]
        .drop(columns=ANNOTATOR_COLUMN)
        .applymap(literal_eval)
        .applymap(tuple)
    )

    new_df["tokens"] = new_df["tokens"].apply(tokens_cleaner)

    new_df["ac"], _ = convert_ac_to_ids(new_df["ac"])

    return new_df


kiran_df = get_annotator_df("krn_common")

sharmila_df = get_annotator_df("shr_common")


merged_df = pd.merge(
    kiran_df, sharmila_df, how=JOIN_TYPE, on="tokens", suffixes=("_k", "_s")
)

aspect_df = merged_df[["ac_k", "ac_s"]]

if JOIN_TYPE == "outer":

    def replace_nan_with_list_nan(own_key: str, other_key: str):
        na_index = aspect_df[own_key].isna()

        new_ac = aspect_df.loc[na_index, other_key].apply(lambda ac: [np.nan] * len(ac))

        aspect_df.loc[na_index, own_key] = new_ac

    replace_nan_with_list_nan("ac_k", "ac_s")
    replace_nan_with_list_nan("ac_s", "ac_k")

assert (
    aspect_df.ac_k.apply(len) != aspect_df.ac_s.apply(len)
).sum() == 0, "After: Len Not Equals Number"

reliability_data = np.stack(
    (
        np.hstack(aspect_df.ac_k.tolist()),
        np.hstack(aspect_df.ac_s.tolist()),
    )
)

alpha = krippendorff.alpha(reliability_data, level_of_measurement="nominal")

# Aspects Alpha: 0.6689491100807659 for both inner and outer join
print(f"Aspects Alpha: {alpha} for {JOIN_TYPE} join")
