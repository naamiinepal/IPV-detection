from typing import Any, Optional

from transformers import AutoModelForTokenClassification, BertForTokenClassification

from constants import MODEL_NAME
from datasets import load_metric

from . import BaseModel, TensorDict


class WordModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        dropout_rate: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.metric = load_metric("seqeval")

    def validation_step(
        self, batch: TensorDict, batch_idx: int, dataloader_idx: int = 0
    ):
        labels = batch["labels"]

        model_pred = self(batch)

        self.log("val_loss", model_pred.loss, sync_dist=True)

        predictions = model_pred.logits.argmax(dim=-1)

        true_labels, true_pred = self.postprocess(predictions, labels)

        self.metric.add_batch(predictions=true_pred, references=true_labels)

    def postprocess(self, predictions, labels):
        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.label_names[lab] for lab in label if lab != -100] for label in labels
        ]
        true_predictions = [
            [self.label_names[p] for p, lab in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def validation_epoch_end(self, outputs: Any):
        results = self.metric.compute()

        # Log scalar and complex outputs
        for key, value in results.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    self.log(f"{key}_{inner_key}", float(inner_value))
            else:
                self.log(key, float(value))

    def predict_step(self, batch: TensorDict, batch_idx: int, dataloader_idx: int = 0):
        predictions = self(batch).argmax(dim=-1)

        true_predictions = [
            [self.label_names[p] for p in prediction] for prediction in predictions
        ]

        return true_predictions

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        datamodule = self.trainer.datamodule
        self.num_labels = datamodule.num_labels
        self.label_names = datamodule.label_names

        self.model: BertForTokenClassification = (
            AutoModelForTokenClassification.from_pretrained(
                self.hparams.model_name_or_path, num_labels=self.num_labels
            )
        )
