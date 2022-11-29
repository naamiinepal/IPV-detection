import math
from typing import List, Mapping, Optional, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

from constants import MODEL_NAME

from . import BaseModel, TensorDict


class WordModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        dropout_rate: float = 0.4,
        calc_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.metric = evaluate.load("seqeval")

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        datamodule = self.trainer.datamodule
        self.label_names = datamodule.label_names
        self.tokenizer = datamodule.tokenizer

        self.model: BertForTokenClassification = (
            AutoModelForTokenClassification.from_pretrained(
                self.hparams.model_name_or_path,
                classifier_dropout=self.hparams.dropout_rate,
                num_labels=len(self.label_names),
            )
        )

        if (stage is None or stage == "fit") and self.hparams.calc_bias:
            # Get frequency of each label without sorting them by their freqeuencies
            # Sort the frequencies by the labels
            # Skip the first -100 label
            count_series = (
                pd.value_counts(
                    np.hstack(datamodule.train_dataset["labels"]), sort=False
                )
                .sort_index()
                .iloc[1:]
            )
            bias_initializer = np.log(count_series) - math.log(sum(count_series))

            self.model.classifier.bias = torch.nn.Parameter(
                torch.as_tensor(bias_initializer, dtype=torch.float32)
            )

    def validation_step(
        self, batch: TensorDict, batch_idx: int, dataloader_idx: int = 0
    ):
        labels = batch["labels"]

        model_pred = self(batch)

        self.log("val_loss", model_pred.loss, sync_dist=True, prog_bar=True)

        predictions = model_pred.logits.argmax(dim=-1)

        true_labels, true_pred = self.postprocess(predictions, labels)

        return {
            "true_labels": true_labels,
            "true_pred": true_pred,
        }

    def postprocess(self, predictions, labels):
        # Remove ignored index (special tokens) and convert to labels
        true_labels = tuple(
            tuple(self.label_names[lab] for lab in label if lab != -100)
            for label in labels
        )
        true_predictions = tuple(
            tuple(
                self.label_names[p] for p, lab in zip(prediction, label) if lab != -100
            )
            for prediction, label in zip(predictions, labels)
        )
        return true_labels, true_predictions

    def validation_epoch_end(self, outputs: List[Mapping[str, Tuple[Tuple]]]):

        if self.trainer.is_global_zero:
            for output in outputs:
                self.metric.add_batch(
                    predictions=output["true_pred"], references=output["true_labels"]
                )

            results = self.metric.compute()

            # Log scalar and complex outputs
            for key, value in results.items():
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        self.log(
                            f"{key}_{inner_key}",
                            float(inner_value),
                            sync_dist=True,
                            rank_zero_only=True,
                        )
                else:
                    self.log(key, float(value), rank_zero_only=True, sync_dist=True)

    def predict_step(self, batch: TensorDict, batch_idx: int, dataloader_idx: int = 0):
        output: TokenClassifierOutput = self(batch)
        predictions = output.logits.argmax(dim=-1)

        true_predictions = [
            [self.label_names[p] for p in prediction] for prediction in predictions
        ]

        splitted_texts = [
            self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
            for ids in batch["input_ids"]
        ]

        return true_predictions, splitted_texts
