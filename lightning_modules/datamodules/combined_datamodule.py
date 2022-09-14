import math
import os.path
from typing import Optional

import numpy as np
import pandas as pd

from constants import MAX_PROC
from datasets import load_dataset, load_from_disk

from .combined_data_collator import CombinedDataCollator
from .word_datamodule import WordDataModule


class CombinedDataModule(WordDataModule):
    def __init__(
        self,
        dataset_path: str,
        val_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.data_collator = CombinedDataCollator(tokenizer=self.tokenizer)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage is None or stage == "fit":

            dataset_path = self.hparams.dataset_path
            tokenized_path = (
                f"{dataset_path}_{self.hparams.model_name_or_path.replace('/', '_')}"
            )
            if os.path.isdir(tokenized_path):
                dataset_full = load_from_disk(tokenized_path)
            else:
                if os.path.isdir(dataset_path):
                    raw_dataset = load_from_disk(dataset_path)
                else:
                    raw_dataset = (
                        load_dataset(
                            "csv",
                            data_files=f"{dataset_path}.csv",
                            split="train",
                        )
                        .remove_columns("annotator")
                        .rename_column("ipv", "sent_label")
                        .train_test_split(test_size=self.hparams.val_ratio)
                        .map(
                            lambda row: {
                                "ac": eval(row["ac"]),
                                "tokens": eval(row["tokens"]),
                            },
                            num_proc=MAX_PROC,
                        )
                    )
                    raw_dataset.save_to_disk(dataset_path)

                dataset_full = raw_dataset.map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    batch_size=64,
                    remove_columns=["ac", "tokens"],
                    num_proc=MAX_PROC,
                )

                dataset_full.save_to_disk(tokenized_path)

            train_dataset = dataset_full["train"]

            sent_pos = sum(train_dataset["sent_label"])
            sent_neg = len(train_dataset) - sent_pos

            self.sent_bias_init = math.log(sent_pos) - math.log(sent_neg)

            count_series = pd.value_counts(
                np.hstack(train_dataset["labels"]), sort=False
            ).sort_index()[1:]
            self.word_bias_init = np.log(count_series) - math.log(sum(count_series))

            self.train_dataset = train_dataset
            self.val_dataset = dataset_full["test"]
