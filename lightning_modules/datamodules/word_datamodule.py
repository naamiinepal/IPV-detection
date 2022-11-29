import json
import os.path
from typing import Iterator, Optional, Sequence

from transformers import DataCollatorForTokenClassification

from constants import MAX_PROC
from datasets import load_dataset, load_from_disk

from . import BaseDataModule


class WordDataModule(BaseDataModule):
    def __init__(
        self,
        dataset_path: str,
        val_ratio: float = 0.1,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )

        with open(os.path.join(dataset_path, "label_names.json")) as f:
            self.label_names = json.load(f)

    def setup(self, stage: Optional[str] = None):
        dataset_path = self.hparams.dataset_path
        tokenized_path = os.path.join(
            f"{dataset_path}_word_cache",
            self.hparams.model_name_or_path.replace("/", "_"),
        )

        if stage is None or stage == "fit" or stage == "validate":
            if self.hparams.use_cache and os.path.isdir(tokenized_path):
                dataset_full = load_from_disk(tokenized_path)
            else:
                dataset_full = (
                    load_dataset(
                        "csv",
                        data_files=os.path.join(dataset_path, "combined.csv"),
                        split="train",
                    )
                    .train_test_split(test_size=self.hparams.val_ratio)
                    .map(
                        lambda row: {
                            "ac": eval(row["ac"]),
                            "tokens": eval(row["tokens"]),
                        },
                        num_proc=MAX_PROC,
                    )
                    .map(
                        self.tokenize_and_align_labels,
                        batched=True,
                        batch_size=1024,
                        remove_columns=["ac", "tokens"],
                        num_proc=MAX_PROC,
                    )
                )

                dataset_full.save_to_disk(tokenized_path)

            self.val_dataset = dataset_full["test"]

            if stage != "validate":
                self.train_dataset = dataset_full["train"]

            # count_series = pd.value_counts(
            #     np.hstack(train_dataset["labels"]), sort=False
            # ).sort_index()[1:]

        if stage is None or stage == "predict":
            if self.hparams.use_cache and os.path.isdir(tokenized_path):
                self.dataset_full = load_from_disk(tokenized_path)
            else:
                # Needed for writing to the predictions file
                self.dataset_full = load_dataset(
                    "csv",
                    data_files=os.path.join(dataset_path, "combined.csv"),
                    split="train",
                ).map(
                    self.pred_tokenize,
                    batched=True,
                    batch_size=1024,
                    num_proc=self.num_workers,
                )

                self.dataset_full.save_to_disk(tokenized_path)

            self.pred_dataset = self.dataset_full.remove_columns("text")

    def pred_tokenize(self, examples: dict):
        tokenized_inputs = self.tokenizer(examples["text"], truncation=True)
        return tokenized_inputs

    @staticmethod
    def align_labels_with_tokens(
        labels: Sequence[int], word_ids: Iterator[Optional[int]]
    ):
        """
        Adding the special tokens [CLS] and [SEP] and subword tokenization creates
            a mismatch between the input and labels.
        A single word corresponding to a single label may be split into two subwords.

        You will need to realign the tokens and labels by:
        1. Mapping all tokens to their corresponding word with the `word_ids` method.
        2. Assigning the label -100 to the special tokens `[CLS]` and `[SEP]`
            so the PyTorch loss function ignores them.
        3. Only labeling the first token of a given word.
            Assign `-100` to other subtokens from the same word.
        """
        new_labels = []
        for word_id in word_ids:
            if word_id is None:
                # Special token
                label = -100
            else:
                label = labels[word_id]
                # Same word as previous token
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
            new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self, examples: dict):
        # Since the input has already been split into words,
        # set is_split_into_words=True to tokenize the words into subwords
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ac"]
        all_word_ids = map(tokenized_inputs.word_ids, range(len(all_labels)))

        new_labels = [
            self.align_labels_with_tokens(*labels_word_ids)
            for labels_word_ids in zip(all_labels, all_word_ids)
        ]

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
