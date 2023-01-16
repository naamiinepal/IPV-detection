from typing import Mapping

import pytorch_lightning as pl
import torch

TensorDict = Mapping[str, torch.Tensor]


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        plateu_factor: float = 0.1,
        plateu_patience: int = 2,
        monitor: str = "val_loss",
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

    def configure_optimizers(self):
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
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.hparams.plateu_factor,
                patience=self.hparams.plateu_patience,
            ),
            "monitor": self.hparams.monitor,
        }
        return [optimizer], [scheduler]
