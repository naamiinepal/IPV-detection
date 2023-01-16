import json
import os.path
from ast import literal_eval
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from transformers import DataCollatorForTokenClassification

from datasets import load_from_disk

from . import BaseDataModule


class WordDataModule(BaseDataModule):
    def __init__(
        self,
        dataset_path: str,
        val_ratio: float = 0.1,
        use_cache: bool = True,
        current_fold: Optional[int] = None,
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
        tokenized_path = self.get_tokenized_path("word")

        kfold_dir = os.path.join(
            self.hparams.dataset_path, "folds", str(self.hparams.current_fold)
        )

        if stage is None or stage == "fit" or stage == "validate":
            if self.hparams.use_cache and os.path.isdir(tokenized_path):
                dataset_full = load_from_disk(tokenized_path)
            else:
                dataset_full = (
                    self.get_raw_dataset_full(kfold_dir)
                    .map(
                        self.eval_ac_tokens,
                        num_proc=self.num_workers,
                    )
                    .map(
                        self.tokenize_and_align_labels,
                        batched=True,
                        batch_size=1024,
                        remove_columns=["ac", "tokens"],
                        num_proc=self.num_workers,
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
            self.assign_pred_dataset(tokenized_path, kfold_dir)

    @staticmethod
    def align_labels_with_tokens(
        labels: Sequence[int], word_ids: Iterable[Optional[int]]
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
        current_word = None
        for word_id in word_ids:
            if word_id is None:
                # Special token
                label = -100
            else:
                label = labels[word_id]
                if word_id != current_word:
                    # Start of a new word!
                    current_word = word_id
                elif label % 2 == 1:
                    # Same word as previous token and
                    #   if the label is B-XXX we change it to I-XXX
                    label += 1

            new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self, examples: Mapping[str, Any]):
        # Since the input has already been split into words,
        # set is_split_into_words=True to tokenize the words into subwords
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ac"]

        # Index of every token from the original texts
        # Example: [None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
        all_word_ids = map(tokenized_inputs.word_ids, range(len(all_labels)))

        new_labels = [
            self.align_labels_with_tokens(*labels_word_ids)
            for labels_word_ids in zip(all_labels, all_word_ids)
        ]

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    @staticmethod
    def eval_ac_tokens(row: Mapping[str, Any]) -> Mapping[str, Union[list, tuple]]:
        return {
            "ac": literal_eval(row["ac"]),
            "tokens": literal_eval(row["tokens"]),
        }
