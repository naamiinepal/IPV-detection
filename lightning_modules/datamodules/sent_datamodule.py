import os.path
from typing import Optional

from transformers import DataCollatorWithPadding

from constants import DATASET_FOLDER
from datasets import load_dataset, load_from_disk

from . import BaseDataModule


class SentDataModule(BaseDataModule):

    labels = {"abuse", "sexual_content_score"}

    def __init__(
        self,
        dataset_path: str = DATASET_FOLDER,
        val_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage is None or stage == "fit":

            dataset_path = self.hparams.dataset_path

            tokenized_path = os.path.join(
                f"{dataset_path}_cache",
                f"{self.hparams.model_name_or_path.replace('/', '_')}",
            )
            if os.path.isdir(tokenized_path):
                dataset_full = load_from_disk(tokenized_path)
            else:
                dataset_full = (
                    load_dataset(
                        "csv",
                        data_files=os.path.join(dataset_path, "combined.csv"),
                        split="train",
                    )
                    .train_test_split(test_size=self.hparams.val_ratio)
                    .map(
                        lambda batch: self.tokenizer(batch["text"], truncation=True),
                        batched=True,
                        batch_size=1024,
                        remove_columns="text",
                        num_proc=self.num_workers,
                    )
                )

                dataset_full.save_to_disk(tokenized_path)

            self.train_dataset = dataset_full["train"]
            self.val_dataset = dataset_full["test"]
