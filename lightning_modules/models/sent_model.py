import math
from typing import Optional

import torch
from constants import MODEL_NAME
from datamodules.sent_datamodule import SentDataModule
from datasets import Dataset
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, AveragePrecision, F1Score, R2Score
from transformers import AutoModel, BertModel

from models import BaseModel

from . import TensorDict


class SentModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        dropout_rate: float = 0.4,
        sexual_weight: float = 0.1,
        calc_bias: bool = False,
        plateu_factor: float = 0.1,
        plateu_patience: int = 2,
        monitor: str = "val_loss",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.model: BertModel = AutoModel.from_pretrained(model_name_or_path)
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.config.hidden_size, 2, bias=not calc_bias),
        )

        self.train_acc = Accuracy(task="binary")
        self.train_auc = AveragePrecision(task="binary")
        self.train_f1score = F1Score(task="binary")
        self.train_r2score = R2Score()

        self.val_acc = Accuracy(task="binary")
        self.val_auc = AveragePrecision(task="binary")
        self.val_f1score = F1Score(task="binary")
        self.val_r2score = R2Score()

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        if (stage is None or stage == "fit") and self.hparams.calc_bias:
            train_dataset: Dataset = self.trainer.datamodule.train_dataset
            pos = sum(train_dataset["abuse"])
            neg = len(train_dataset) - pos

            abuse_initializer = math.log(pos) - math.log(neg)

            score = tuple(
                sc for sc in train_dataset["sexual_content_score"] if sc is not None
            )
            sexual_initializer = sum(score) / len(score)

            # Add bias to the model
            self.classification_head[-1].bias = nn.Parameter(
                torch.tensor(
                    (abuse_initializer, sexual_initializer),
                    dtype=torch.float32,
                    device=self.device,
                )
            )

    def forward(self, inputs: TensorDict):
        pooler_output = self.model(
            **{
                key: val
                for key, val in inputs.items()
                if key not in SentDataModule.labels
            }
        ).pooler_output
        logits = self.classification_head(pooler_output)
        return logits

    def training_step(self, batch: TensorDict, batch_idx: int):
        logits = self(batch)

        abuse_pred = logits[:, 0]
        abuse_target = batch["abuse"]

        self.train_acc(abuse_pred, abuse_target)
        self.log("train_acc", self.train_acc)

        self.train_auc(abuse_pred, abuse_target)
        self.log("train_auc", self.train_auc)

        self.train_f1score(abuse_pred, abuse_target)
        self.log("train_f1score", self.train_f1score)

        binary_abuse = abuse_target.bool()  # mask for abuse class
        sexual_target = batch["sexual_content_score"][binary_abuse]
        sexual_pred = logits[:, 1][binary_abuse]

        self.train_r2score(sexual_pred, sexual_target)
        self.log("train_r2score", self.train_r2score)

        abuse_loss, sexual_loss, loss = self.calculate_loss(
            abuse_pred, abuse_target, sexual_pred, sexual_target
        )

        self.log_dict(
            {
                "train_abuse_loss": abuse_loss,
                "train_sexual_loss": sexual_loss,
                "train_loss": loss,
            }
        )

        return loss

    def validation_step(
        self, batch: TensorDict, batch_idx: int, dataloader_idx: int = 0
    ):
        logits = self(batch)

        abuse_pred = logits[:, 0]
        abuse_target = batch["abuse"]

        self.val_acc(abuse_pred, abuse_target)
        self.log("val_acc", self.val_acc)

        self.val_auc(abuse_pred, abuse_target)
        self.log("val_auc", self.val_auc)

        self.val_f1score(abuse_pred, abuse_target)
        self.log("val_f1score", self.val_f1score, prog_bar=True)

        binary_abuse = abuse_target.bool()  # mask for abuse class
        sexual_target = batch["sexual_content_score"][binary_abuse]
        sexual_pred = logits[:, 1][binary_abuse]

        self.val_r2score(sexual_pred, sexual_target)
        self.log("val_r2score", self.val_r2score)

        abuse_loss, sexual_loss, loss = self.calculate_loss(
            abuse_pred, abuse_target, sexual_pred, sexual_target
        )

        self.log_dict(
            {
                "val_abuse_loss": abuse_loss,
                "val_sexual_loss": sexual_loss,
                "val_loss": loss,
            },
            sync_dist=True,
            prog_bar=True,
        )

    def calculate_loss(
        self,
        abuse_pred: torch.Tensor,
        abuse_target: torch.Tensor,
        sexual_pred: torch.Tensor,
        sexual_target: torch.Tensor,
    ):
        abuse_loss = F.binary_cross_entropy_with_logits(
            abuse_pred, abuse_target.float()
        )

        sexual_loss = F.mse_loss(sexual_pred, sexual_target.float())

        loss = abuse_loss + self.hparams.sexual_weight * sexual_loss

        return abuse_loss, sexual_loss, loss

    def predict_step(self, batch: TensorDict, batch_idx: int, dataloader_idx: int = 0):
        logits: torch.Tensor = self(batch)

        return torch.sigmoid(logits[:, 0]), logits[:, 1]
