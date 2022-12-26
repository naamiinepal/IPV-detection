import os.path

import krippendorff
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("datasets", "sentence")


def get_df(annotator: str):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{annotator}.csv"), low_memory=False)
    df = df[df["annotator"] == "common"]
    df["annotator"] = annotator
    df["text"] = df["text"].str.strip().str.normalize("NFKC")
    return df


kiran_df = get_df("Kiran")
sharmila_df = get_df("Sharmila")

merged_df = pd.merge(kiran_df, sharmila_df, on="text", suffixes=("_k", "_s"))

abuse_reliability_data = np.stack((merged_df["abuse_k"], merged_df["abuse_s"]))

abuse_alpha = krippendorff.alpha(abuse_reliability_data, level_of_measurement="nominal")

# Abuse Alpha 0.7849921124336727
print("Abuse Alpha", abuse_alpha)

sexual_score_reliability_data = np.stack(
    (merged_df["sexual_content_score_k"], merged_df["sexual_content_score_s"])
)

sexual_score_alpha = krippendorff.alpha(
    sexual_score_reliability_data, level_of_measurement="ordinal"
)

# Sexual Score Alpha 0.7090561945829004
print("Sexual Score Alpha", sexual_score_alpha)
