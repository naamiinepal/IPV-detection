#! .venv/bin/python

import os.path
from argparse import ArgumentParser, BooleanOptionalAction, Namespace

import pandas as pd

from datamodules.word_datamodule import WordDataModule
from models.word_model import WordModel
import pytorch_lightning as pl


def main(args: Namespace):

    DATA_DIR = os.path.join("datasets", args.dataset)

    dm = WordDataModule.load_from_checkpoint(
        args.ckpt_path,
        dataset_path=DATA_DIR,
        batch_size=args.batch_size,
        pinmemory=False,
        use_cache=args.use_cache,
    )

    model: WordModel = WordModel(dm.hparams.model_name_or_path)

    trainer = pl.Trainer(
        accelerator="gpu" if args.gpu >= 0 else "cpu",
        devices=[args.gpu] if args.gpu >= 0 else None,
        logger=False,
    )

    predictions = trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_path)

    aspects = []
    splitted_texts = []

    for pred, tokens in predictions:
        splitted_texts.extend(tokens)
        aspects.extend(pred)

    pred_df = pd.DataFrame(
        {
            "text": dm.dataset_full["text"],
            "splitted_texts": splitted_texts,
            "aspects": aspects,
        }
    )

    pred_df.to_csv(os.path.join(DATA_DIR, "predictions.csv"), index=False)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/word_muril/epoch=54-val_loss=0.968.ckpt",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="raw_words",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="which GPU to use (negative means CPU)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size for dataloader"
    )
    parser.add_argument(
        "--use_cache",
        default=True,
        help="Whether to use the tokenized cache",
        action=BooleanOptionalAction,
    )
    args = parser.parse_args()
    main(args)
