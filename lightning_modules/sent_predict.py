#! .venv/bin/python

import os.path
from argparse import ArgumentParser, BooleanOptionalAction, Namespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datamodules.sent_datamodule import SentDataModule
from models.sent_model import SentModel


def main(args: Namespace):

    device = torch.device("cuda", args.gpu) if args.gpu >= 0 else torch.device("cpu")

    model: SentModel = (
        SentModel.load_from_checkpoint(args.ckpt_path, calc_bias=False)
        .eval()
        .to(device)
    )

    DATA_DIR = os.path.join("datasets", "raw")

    dm = SentDataModule.load_from_checkpoint(
        args.ckpt_path,
        dataset_path=DATA_DIR,
        batch_size=args.batch_size,
        use_cache=args.use_cache,
    )

    dm.setup("predict")

    dl = dm.predict_dataloader()

    abuse_pred_list = []
    sexual_score_list = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dl)):
            batch = batch.to(device)
            abuse_pred, sexual_score = model.predict_step(batch, batch_idx=i)
            abuse_pred_list.append(abuse_pred.cpu().numpy())
            sexual_score_list.append(sexual_score.cpu().numpy())

    pred_df = pd.DataFrame(
        {
            "text": dm.dataset_full["text"],
            "abuse_pred": np.concatenate(abuse_pred_list),
            "sexual_score": np.concatenate(sexual_score_list),
        }
    )

    pred_df.to_csv(os.path.join(DATA_DIR, "predictions.csv"), index=False)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=(
            "checkpoints/sent_muril_bias_plateu_norm_free"
            "/epoch=04-val_loss=0.982.ckpt"
        ),
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
