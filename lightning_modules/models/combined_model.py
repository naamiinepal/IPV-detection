from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from constants import MODEL_NAME
from datasets import load_metric
from torch import nn
from transformers import AutoModel

from .word_model import WordModel


class CombinedModel(WordModel):
    _removable_keys = {"labels", "sent_label"}

    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        lambda_weight: float = 1,
        word_dropout_rate: float = 0.4,
        sent_dropout_rate: float = 0.4,
        calc_word_bias: bool = True,
        calc_sent_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained(model_name_or_path)

        self.metric = load_metric("seqeval")

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
            nn.Sequential(nn.Dropout(dropout_rate), linear)
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

        true_labels, true_predictions = self.postprocess(predictions, labels)
        self.metric.add_batch(predictions=true_predictions, references=true_labels)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        word_logits, sent_logits = self(**batch)

        word_pred = word_logits.argmax(dim=-1)

        sent_pred = F.sigmoid(sent_logits.squeeze())

        true_word_pred = [[self.label_names[p] for p in pred] for pred in word_pred]

        return true_word_pred, sent_pred

    def get_word_loss(self, word_logits, labels):
        return F.cross_entropy(word_logits.view(-1, self.num_labels), labels.view(-1))

    @staticmethod
    def get_sent_loss(sent_logits, sent_label):
        return F.binary_cross_entropy_with_logits(
            sent_logits.view(-1), sent_label.float()
        )

    def get_total_loss(self, word_loss, sent_loss):
        return word_loss + self.hparams.lambda_weight * sent_loss

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        if stage is None or stage == "fit":
            datamodule = self.trainer.datamodule

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
