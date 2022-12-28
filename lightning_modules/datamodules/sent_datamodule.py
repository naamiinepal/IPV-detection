import os.path
from typing import Optional

from transformers import DataCollatorWithPadding

from datasets import load_dataset, load_from_disk

from . import BaseDataModule


class SentDataModule(BaseDataModule):

    labels = {"abuse", "sexual_content_score"}

    def __init__(
        self,
        dataset_path: str,
        val_ratio: float = 0.1,
        use_cache: bool = True,
        current_fold: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], truncation=True)

    def setup(self, stage: Optional[str] = None):
        dataset_path = self.hparams.dataset_path

        tokenized_root = f"{dataset_path}_sent_cache"

        current_fold = self.hparams.current_fold
        if current_fold is not None:
            tokenized_root = f"{tokenized_root}_fold_{current_fold}"

        tokenized_path = os.path.join(
            tokenized_root,
            self.hparams.model_name_or_path.replace("/", "_"),
        )

        kfold_dir = os.path.join(dataset_path, "folds", str(current_fold))

        # Assign train/val datasets for use in dataloaders
        if stage is None or stage == "fit" or stage == "validate":
            if self.hparams.use_cache and os.path.isdir(tokenized_path):
                dataset_full = load_from_disk(tokenized_path)
            else:
                if current_fold is None:
                    dataset_full = load_dataset(
                        "csv",
                        data_files=os.path.join(dataset_path, "combined.csv"),
                        split="train",
                    ).train_test_split(test_size=self.hparams.val_ratio)
                else:
                    dataset_full = load_dataset("csv", data_dir=kfold_dir)

                dataset_full = dataset_full.map(
                    self.tokenize,
                    batched=True,
                    batch_size=1024,
                    remove_columns="text",
                    num_proc=self.num_workers,
                )

                dataset_full.save_to_disk(tokenized_path)

            self.val_dataset = dataset_full["test"]

            if stage != "validate":
                self.train_dataset = dataset_full["train"]

        if stage is None or stage == "predict":
            if self.hparams.use_cache and os.path.isdir(tokenized_path):
                self.dataset_full = load_from_disk(tokenized_path)
            else:
                # Needed for writing to the predictions file
                self.dataset_full = load_dataset(
                    "csv",
                    data_files=os.path.join(dataset_path, "combined.csv")
                    if current_fold is None
                    else os.path.join(kfold_dir, "test.csv"),
                    split="train",  # Default is train for every loading
                ).map(
                    self.tokenize,
                    batched=True,
                    batch_size=1024,
                    num_proc=self.num_workers,
                )

                self.dataset_full.save_to_disk(tokenized_path)

            self.pred_dataset = self.dataset_full.remove_columns("text")
