from typing import Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


class CombinedDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features):

        sent_label = tuple(feature.pop("sent_label") for feature in features)

        # Conversion to tensors will fail if we have labels
        # as they are not of the same length yet.
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
        )

        labels = tuple(tuple(feature["labels"]) for feature in features)

        sequence_length = len(batch["input_ids"][0])

        paddings = tuple((-100,) * (sequence_length - len(label)) for label in labels)

        batch["labels"] = (
            [label + padding for label, padding in zip(labels, paddings)]
            if self.tokenizer.padding_side == "right"
            else [padding + label for label, padding in zip(labels, paddings)]
        )

        batch["sent_label"] = sent_label

        # Convert to tensor now
        batch = {k: torch.as_tensor(v, dtype=torch.int64) for k, v in batch.items()}

        return batch
