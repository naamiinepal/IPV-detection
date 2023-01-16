import os.path
from typing import Optional

from datasets import load_from_disk
from transformers import DataCollatorWithPadding

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

    def setup(self, stage: Optional[str] = None):
        tokenized_path = self.get_tokenized_path("sent")

        kfold_dir = os.path.join(
            self.hparams.dataset_path, "folds", str(self.hparams.current_fold)
        )

        # Assign train/val datasets for use in dataloaders
        if stage is None or stage == "fit" or stage == "validate":
            if self.hparams.use_cache and os.path.isdir(tokenized_path):
                dataset_full = load_from_disk(tokenized_path)
            else:
                dataset_full = self.get_raw_dataset_full(kfold_dir).map(
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
            self.assign_pred_dataset(tokenized_path, kfold_dir)
