import os

import numpy as np
from sklearn.model_selection import KFold

from preprocess_word import processed_word_df

DATA_DIR = os.path.join("datasets", "word")

SEED = 4242

FOLDS = 5

df = processed_word_df(save_df=False, save_label_names=False)

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

FOLD_DIR = os.path.join(DATA_DIR, "folds")

os.makedirs(FOLD_DIR, exist_ok=True)


def save_fold_data(index: np.ndarray, is_train: bool):
    fold_data = df.iloc[index]

    filename = "train" if is_train else "test"

    fold_data.to_csv(
        os.path.join(CURRENT_FOLD_DIR, f"{filename}.csv"),
        index=False,
    )


for fold, (train_index, test_index) in enumerate(kf.split(df), start=1):
    print(f"Fold {fold}")
    print(f"\t Train: index shape={train_index.shape}")
    print(f"\t Test: index shape={test_index.shape}")

    CURRENT_FOLD_DIR = os.path.join(FOLD_DIR, str(fold))

    os.makedirs(CURRENT_FOLD_DIR, exist_ok=True)

    save_fold_data(train_index, is_train=True)
    save_fold_data(test_index, is_train=False)
