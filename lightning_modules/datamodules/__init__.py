import os.path

import pytorch_lightning as pl
from constants import MAX_PROC, MODEL_NAME
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        batch_size: int = 64,
        max_workers: int = MAX_PROC,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.num_workers = min(os.cpu_count(), max_workers)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.data_collator,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def get_tokenized_path(self, dm_type: str) -> str:
        """Get tokenized path from the dataset path and current fold

        Args:
            dm_type (str): The datamodule type. Typically: `word` or `sent`

        Returns:
            str: The path for the tokenized cache folder for the current fold
        """
        tokenized_root = f"{self.hparams.dataset_path}_{dm_type}_cache"

        current_fold = self.hparams.current_fold
        if current_fold is not None:
            tokenized_root = f"{tokenized_root}_fold_{current_fold}"

        return os.path.join(
            tokenized_root,
            self.hparams.model_name_or_path.replace("/", "_"),
        )

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], truncation=True)

    def get_raw_dataset_full(self, kfold_dir: str) -> DatasetDict:
        """Get the dataset for the training and validation

        Args:
            kfold_dir (str): The directory containing the current fold
                train and val data

        Returns:
            DatasetDict: The dataset dict with train and test as the keys
        """

        return (
            load_dataset(
                "csv",
                data_files=os.path.join(self.hparams.dataset_path, "combined.csv"),
                split="train",
            ).train_test_split(test_size=self.hparams.val_ratio)
            if self.hparams.current_fold is None
            else load_dataset("csv", data_dir=kfold_dir)
        )

    def assign_pred_dataset(self, tokenized_path: str, kfold_dir: str):
        """Assign prediction dataset to approproate value

        Args:
            tokenized_path (str): The path to the cache root for tokenizer
            kfold_dir (str): The folder containing the current fold `test.csv`
        """

        if self.hparams.use_cache and os.path.isdir(tokenized_path):
            self.dataset_full = load_from_disk(tokenized_path)
        else:
            # Needed for writing to the predictions file
            self.dataset_full = load_dataset(
                "csv",
                data_files=os.path.join(self.hparams.dataset_path, "combined.csv")
                if self.hparams.current_fold is None
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
