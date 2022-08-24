import os.path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from constants import MAX_PROC, MODEL_NAME


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
