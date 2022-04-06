# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 07:31:31 2022

@author: Sagun Shakya
"""

# Libraries.
from os.path import join, exists
import os
from random import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from time import time

import pandas as pd
import numpy as np
np.random.seed(100)

from warnings import filterwarnings
filterwarnings(action='ignore')

# Local Modules.
from models.ml_models import ml_utils
from .pipeline import MLPipeline

def main(args, logger):
    logger.info("Running Adaboost Classifier...\n")
    
    # Num of k-folds.
    k_fold = args.kfold

    # Record for storing train, val and test metrics.
    train_columns = ['train_acc', 'train_pr', 'train_rec', 'train_f1', 'train_auc']
    val_columns = ['val_acc', 'val_pr', 'val_rec', 'val_f1', 'val_auc']
    columns = train_columns + val_columns

    record = np.zeros((k_fold, len(columns)))

    # Start the main loop from here for a given 'k'.
    for ii in range(k_fold):

        # Path to data files.
        k = ii + 1

        pipe = MLPipeline(args, logger, k)

        # Load vectors.
        x_train, x_val, y_train, y_val = pipe.load_vectors()

        start = time()
        logger.info(f'\n Run program: {ml_utils.current_timestamp()}\n')

        # Initialize model.
        model = AdaBoostClassifier(n_estimators = args.adaboost.n_estimators, 
                                    learning_rate = args.adaboost.learning_rate,
                                    random_state = args.adaboost.random_state)
        
        # Make predictions.
        y_pred_train, y_pred_val = pipe.fit_predict(model = model)
    
        # Evaluation.
        train_acc, train_pr, train_rec, train_f1, train_auc = ml_utils.classification_metrics(y_train, y_pred_train)
        val_acc, val_pr, val_rec, val_f1, val_auc = ml_utils.classification_metrics(y_val, y_pred_val)

        end = time()

        logger.info(f'Completed!\nTime Elapsed : {end - start:5.3f} seconds.')

        # Appending to record.
        record[ii] = (
                    train_acc, train_pr, train_rec, train_f1, train_auc,
                    val_acc, val_pr, val_rec, val_f1, val_auc,
                    )
                    
        # Verbose.
        if args.verbose:
            print(f'\nResults for k = {k}:')
            print('-'*50 + '\n')
            ml_utils.verbosity(train_acc, train_pr, train_rec, train_f1, train_auc, logger, mode = 'train')
            ml_utils.verbosity(val_acc, val_pr, val_rec, val_f1, val_auc, logger, mode = 'val')

    record_df = pd.DataFrame(record, index = np.arange(1, k + 1), columns = columns)
    record_df = record_df.append([record_df.mean(), record_df.std()], ignore_index=True)
    
    # Average values.
    logger.info(f'\nAverage Values:\n{record_df.iloc[:-2, :].mean(axis = 0).round(3)}\n')

    # To csv.
    cache_path = join(args.cache_dir, 'adaboost')
    if not exists(cache_path):
        os.mkdir(cache_path)
        
    record_df.to_csv(join(cache_path, f'adeaboost_{str(args.vectorizer.mode)}_{str(args.vectorizer.max_features)}_{str(args.adaboost.n_estimators)}_{str(args.adaboost.learning_rate)}_results.csv'))
    logger.info("Results Summary: \n")
    logger.info(f"{record_df.round(3)}")

