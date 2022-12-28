import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_DIR = os.path.join("datasets", "sentence")

SEED = 4242

USE_COLS = ("text", "abuse", "sexual_content_score")
FOLDS = 5


def get_df(annotator: str):
    df = pd.read_csv(
        os.path.join(DATA_DIR, f"{annotator}.csv"), low_memory=False, usecols=USE_COLS
    )

    # Normalize texts
    df["text"] = df["text"].str.strip().str.normalize("NFKC")

    # Make sexual score annotation more uniform
    is_abuse = df["abuse"].astype(bool)
    df.loc[~is_abuse, "sexual_content_score"] = pd.NA

    # Clip the annotation between 1 and 10 (inclusive)
    df.loc[is_abuse, "sexual_content_score"].clip(1, 10, inplace=True)

    return df


sharmila_df = get_df("Sharmila")
kiran_df = get_df("Kiran")

# Keep in this order to compare with the previously generated combined.csv
combined_df = pd.concat((kiran_df, sharmila_df)).drop_duplicates(ignore_index=True)

# Cannot pass None type to make a batch of tensors
combined_df["sexual_content_score"].fillna(0, inplace=True)

y = combined_df["abuse"]
X = np.zeros_like(y)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

FOLD_DIR = os.path.join(DATA_DIR, "folds")

os.makedirs(FOLD_DIR, exist_ok=True)


def save_fold_data(index: np.ndarray, is_train: bool):
    fold_data = combined_df.iloc[index]

    filename = "train" if is_train else "test"

    fold_data.to_csv(
        os.path.join(CURRENT_FOLD_DIR, f"{filename}.csv"),
        index=False,
    )


for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
    print(f"Fold {fold}")
    print(f"\t Train: index shape={train_index.shape}")
    print(f"\t Test: index shape={test_index.shape}")

    CURRENT_FOLD_DIR = os.path.join(FOLD_DIR, str(fold))

    os.makedirs(CURRENT_FOLD_DIR, exist_ok=True)

    save_fold_data(train_index, is_train=True)
    save_fold_data(test_index, is_train=False)
