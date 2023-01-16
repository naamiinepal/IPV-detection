# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 07:31:31 2022

@author: Sagun Shakya
"""

import os
import pickle

# Libraries.
from os.path import exists, join

import numpy as np
import pandas as pd

np.random.seed(100)

from warnings import filterwarnings

filterwarnings(action="ignore")

from dataloader.ml_pipeline import MLPipeline
from models.ml_models.load_model import InstantiateModel

# Local Modules.
from utilities.ml_utils import classification_metrics, current_timestamp, verbosity


class MLTrainer:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info(f"Running {args.model} Classifier...\n")

    def _write_results(self, record_df: pd.DataFrame, cache_path: str):
        """
        Write results to cache_dir.
        """
        suffix = "_".join(
            (str(values) for params, values in self.args[self.args.model].items())
        )
        filename = f"{self.args.model}_{self.args.vectorizer.mode}_{self.args.vectorizer.max_features}_{suffix}_results.csv"
        record_df.to_csv(join(cache_path, filename))
        self.logger.info("Results Summary: \n")
        self.logger.info(f"{record_df.round(3)}")

    def _train_i_fold(self, k: int):
        """
        Train only the i-th fold and return the results for train and validation sets.
        """
        # Pipeline object.
        pipe = MLPipeline(self.args, self.logger, k)

        # Load vectors.
        x_train, x_val, y_train, y_val = pipe.load_vectors()

        self.logger.info(f"\n Run program (Fold {k}): {current_timestamp()}\n")

        # Initialize model.
        model = InstantiateModel(self.args, self.logger).__get__()

        # Make predictions.
        y_pred_train, y_pred_val = pipe.fit_predict(model)

        # Evaluation.
        train_acc, train_pr, train_rec, train_f1, train_auc = classification_metrics(
            y_train, y_pred_train
        )
        val_acc, val_pr, val_rec, val_f1, val_auc = classification_metrics(
            y_val, y_pred_val
        )

        # Return results.
        results = (
            train_acc,
            train_pr,
            train_rec,
            train_f1,
            train_auc,
            val_acc,
            val_pr,
            val_rec,
            val_f1,
            val_auc,
        )

        return results

    def train(self):
        """
        Trains k - folds and saves the results in cache directory.
        """
        # Num of folds.
        k_fold = self.args.kfold

        # Record for storing train, val and test metrics.
        train_columns = ["train_acc", "train_pr", "train_rec", "train_f1", "train_auc"]
        val_columns = ["val_acc", "val_pr", "val_rec", "val_f1", "val_auc"]
        columns = train_columns + val_columns

        record = np.zeros((k_fold, len(columns)))

        # Start the main loop from here for a given 'k'.
        for ii in range(k_fold):

            # Current Fold..
            k = ii + 1

            # Results for i-th fold.
            results = self._train_i_fold(k)

            # Appending to record.
            record[ii] = results

            # Verbose.
            if self.args.verbose:
                print(f"\nResults for k = {k}:")
                print("-" * 50 + "\n")
                train_acc, train_pr, train_rec, train_f1, train_auc = results[:5]
                val_acc, val_pr, val_rec, val_f1, val_auc = results[5:]
                verbosity(
                    train_acc,
                    train_pr,
                    train_rec,
                    train_f1,
                    train_auc,
                    self.logger,
                    mode="train",
                )
                verbosity(
                    val_acc, val_pr, val_rec, val_f1, val_auc, self.logger, mode="val"
                )

        record_df = pd.DataFrame(record, index=np.arange(1, k + 1), columns=columns)
        record_df = record_df.append(
            [record_df.mean(), record_df.std()], ignore_index=True
        )

        # Average values.
        self.logger.info(
            f"\nAverage Values:\n{record_df.iloc[:-2, :].mean(axis = 0).round(3)}\n"
        )

        # To csv.
        cache_path = join(self.args.cache_dir, self.args.model)
        if not exists(cache_path):
            os.mkdir(cache_path)

        # Write Results to cache Directory.
        self._write_results(record_df, cache_path)
