from typing import Mapping, Optional

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

TensorDict = Mapping[str, torch.Tensor]


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

    def forward(self, inputs: TensorDict):
        return self.model(**inputs)

    def training_step(self, batch: TensorDict, batch_idx: int):
        loss = self(batch).loss

        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: TensorDict, batch_idx: int, dataloader_idx: int = 0
    ):
        model_pred = self(**batch)

        self.log("val_loss", model_pred.loss, sync_dist=True)

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            trainer = self.trainer
            datamodule = trainer.datamodule

            # Calculate total steps
            eff_batch_size = datamodule.hparams.batch_size * trainer.num_devices
            self.total_steps = (
                len(datamodule.train_dataset) // eff_batch_size
            ) * trainer.max_epochs

            # Doesn't work for distributed training for now
            # self.total_steps = self.trainer.estimated_stepping_batches
            self.warmup_steps = int(self.total_steps * self.hparams.warmup_ratio)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = {"bias", "LayerNorm.weight"}
        total_parameters = self.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in total_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in total_parameters if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
