import json
import math
import os.path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lightning_modules.data.combined_data_collator import CombinedDataCollator
from constants import DATASET_FOLDER, MAX_PROC, MODEL_NAME


class CombinedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        data_dir: str = DATASET_FOLDER,
        model_name_or_path: str = MODEL_NAME,
        val_ratio: float = 0.1,
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)

        self.data_collator = CombinedDataCollator(tokenizer=self.tokenizer)

        #         self.id2label = self.load_json("id2label")
        self.label_names = self.load_json("label_names")
        #         self.label2id = self.load_json("label2id")

        self.num_labels = len(self.label_names)

        # Assign train/val datasets for use in dataloaders
        if stage is None or stage == "fit":

            dataset_path = os.path.join(
                self.hparams.data_dir, self.hparams.dataset_path
            )
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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=MAX_PROC,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=MAX_PROC,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

    def load_json(self, file_prefix: str):
        with open(os.path.join(self.hparams.data_dir, f"{file_prefix}.json")) as f:
            return json.load(f)

    @staticmethod
    def align_labels_with_tokens(
        labels: Sequence[int], word_ids: Iterator[Optional[int]]
    ):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:
                # Special token
                label = -100
            else:
                label = labels[word_id]
                if word_id != current_word:
                    # Start of a new word!
                    current_word = word_id

                # Same word as previous token
                # If the label is B-XXX we change it to I-XXX
                elif label % 2 == 1:
                    label += 1
            new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self, examples: dict):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ac"]
        all_word_ids = map(tokenized_inputs.word_ids, range(len(all_labels)))

        new_labels = [
            self.align_labels_with_tokens(*labels_word_ids)
            for labels_word_ids in zip(all_labels, all_word_ids)
        ]

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
