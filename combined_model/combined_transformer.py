from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_metric
from torch import nn
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup

from constants import MODEL_NAME


class CombinedTransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        lambda_weight: float = 1,
        word_dropout_rate: float = 0.4,
        sent_dropout_rate: float = 0.4,
        calc_word_bias: bool = True,
        calc_sent_bias: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained(model_name_or_path)

        self.metric = load_metric("seqeval")

        self._removable_keys = {"labels", "sent_label"}

    @staticmethod
    def head_maker(
        input_dim: int,
        output_dim: int,
        dropout_rate: float,
        bias_initializer: Optional[np.ndarray],
    ):
        linear = nn.Linear(input_dim, output_dim, bias=bias_initializer is None)

        # Use better initializer
        if bias_initializer is not None:
            linear.bias = nn.Parameter(
                torch.as_tensor(bias_initializer, dtype=torch.float32)
            )

        return (
            nn.Sequential(linear, nn.Dropout(dropout_rate))
            if dropout_rate > 0
            else linear
        )

    def forward(self, **inputs):
        model_output = self.model(
            **{k: v for k, v in inputs.items() if k not in self._removable_keys}
        )
        word_logits = self.word_head(model_output.last_hidden_state)
        sent_logits = self.sent_head(model_output.pooler_output)

        return word_logits, sent_logits

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        sent_label = batch["sent_label"]

        word_logits, sent_logits = self(**batch)

        word_loss = self.get_word_loss(word_logits, labels)
        sent_loss = self.get_sent_loss(sent_logits, sent_label)

        total_loss = self.get_total_loss(word_loss, sent_loss)

        self.log_dict(
            {
                "train_word_loss": word_loss,
                "train_sent_loss": sent_loss,
                "train_total_loss": total_loss,
            },
            sync_dist=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        labels = batch["labels"]
        sent_label = batch["sent_label"]

        word_logits, sent_logits = self(**batch)

        word_loss = self.get_word_loss(word_logits, labels)
        sent_loss = self.get_sent_loss(sent_logits, sent_label)

        total_loss = self.get_total_loss(word_loss, sent_loss)

        self.log_dict(
            {
                "val_total_loss": total_loss,
                "val_word_loss": word_loss,
                "val_sent_loss": sent_loss,
            },
            sync_dist=True,
        )

        predictions = word_logits.argmax(dim=-1)

        return self.postprocess(predictions, labels)

    def validation_epoch_end(self, outputs):
        for true_labels, true_predictions in outputs:
            self.metric.add_batch(predictions=true_predictions, references=true_labels)
        results = self.metric.compute()

        scalar_dict = {
            k: float(v) for k, v in results.items() if not (isinstance(v, dict))
        }

        complex_dict = {k: v for k, v in results.items() if isinstance(v, dict)}

        for outer_key, outer_value in complex_dict.items():
            for key, value in outer_value.items():
                scalar_dict[f"{outer_key}_{key}"] = float(value)

        self.log_dict(scalar_dict, sync_dist=True)

    def postprocess(self, predictions, labels):
        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.label_names[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [self.label_names[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        word_logits, sent_logits = self(**batch)

        word_pred = word_logits.argmax(dim=-1)

        sent_pred = F.sigmoid(sent_logits.squeeze())
        return word_pred, sent_pred

    def get_word_loss(self, word_logits, labels):
        return F.cross_entropy(word_logits.view(-1, self.num_labels), labels.view(-1))

    @staticmethod
    def get_sent_loss(sent_logits, sent_label):
        return F.binary_cross_entropy_with_logits(
            sent_logits.view(-1), sent_label.float()
        )

    def get_total_loss(self, word_loss, sent_loss):
        return word_loss + self.hparams.lambda_weight * sent_loss

    def setup(self, stage=None):
        if stage is None or stage == "fit":
            datamodule = self.trainer.datamodule

            # Calculate total steps
            eff_batch_size = datamodule.hparams.batch_size * self.trainer.num_devices
            self.total_steps = (
                len(datamodule.train_dataset) // eff_batch_size
            ) * self.trainer.max_epochs
            self.warmup_steps = int(self.total_steps * self.hparams.warmup_ratio)

            self.label_names = datamodule.label_names

            self.num_labels = datamodule.num_labels

            config = self.model.config

            self.word_head = self.head_maker(
                config.hidden_size,
                datamodule.num_labels,
                self.hparams.word_dropout_rate,
                datamodule.word_bias_init if self.hparams.calc_word_bias else None,
            )

            self.sent_head = self.head_maker(
                config.embedding_size,
                1,
                self.hparams.sent_dropout_rate,
                datamodule.sent_bias_init if self.hparams.calc_sent_bias else None,
            )

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = {"bias", "LayerNorm.weight"}
        total_parameters = (
            tuple(self.model.named_parameters())
            + tuple(self.word_head.named_parameters())
            + tuple(self.sent_head.named_parameters())
        )
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
