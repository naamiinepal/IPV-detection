# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 07:31:31 2022

@author: Sagun Shakya
"""

# Libraries.
from os.path import join, exists
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from time import time

import pandas as pd
import numpy as np
np.random.seed(100)

from warnings import filterwarnings
filterwarnings(action='ignore')

# Local Modules.
from models.ml_models import ml_utils

def main(args, logger):
    # Num of k-folds.
    k_fold = args.kfold

    # Record for storing train, val and test metrics.
    train_columns = ['train_acc', 'train_pr', 'train_rec', 'train_f1', 'train_auc']
    val_columns = ['val_acc', 'val_pr', 'val_rec', 'val_f1', 'val_auc']
    columns = train_columns + val_columns

    record = np.zeros((k_fold, len(columns)))

    # Define tokenizer function.
    tokenize = lambda string: string.split()

    # Data Directory.
    data_path = args.data_path    #'.\data\kfold'

    # Start the main loop from here for a given 'k'.
    for ii in range(k_fold):

        # Path to data files.
        k = ii + 1
        file_path = join(data_path, str(k))

        # Load train, val and test data.
        train_df = pd.read_csv(join(file_path, 'train.txt'), header = None)
        val_df = pd.read_csv(join(file_path, 'val.txt'), header = None)
        
        # Name columns.
        train_df.columns = val_df.columns = ('id', 'text', 'pol')

        # Main df.
        df = pd.concat([train_df, val_df])
        df.reset_index(drop = True, inplace = True)

        # Vectorizer.
        if args.svm.vectorizer == "count":
            # For text.
            vectorizer = CountVectorizer(lowercase = False,
                                        ngram_range = (1,2),
                                        max_features= args.svm.max_features,
                                        preprocessor = lambda x: x,
                                        tokenizer = lambda sentence : tokenize(sentence),
                                        encoding = "utf-8")
        
        elif args.svm.vectorizer == "tfidf":
            # For text.
            vectorizer = TfidfVectorizer(lowercase = False,
                                        ngram_range = (1,2),
                                        max_features= args.svm.max_features,
                                        preprocessor = lambda x: x,
                                        tokenizer = lambda sentence : tokenize(sentence),
                                        encoding = "utf-8")
        
        vectorizer.fit(df['text'])
        print(f"\nNumber of 'text' features loaded: {len(vectorizer.get_feature_names_out())}")

        # Input features : Vectorized texts.
        x_train = vectorizer.transform(train_df['text'])
        x_val = vectorizer.transform(val_df['text'])
        x_test = vectorizer.transform(test_df['text'])
        
        # Gold Labels.
        y_train = train_df['pol']
        y_val = val_df['pol']
        y_test = test_df['pol']

        print(f'Train set shape : {x_train.shape}')
        print(f'Val set shape : {x_val.shape}')
        print(f'Test set shape : {x_test.shape}')

        # Initialize SVC.
        start = time()
        logger.info(f'\n Run program: {ml_utils.current_timestamp()}\n')
        svm = SVC(kernel = args.svm.kernel, C = args.svm.C)
        svm.fit(x_train, y_train)

        # Predictions.
        y_pred_train = svm.predict(x_train)
        y_pred_val = svm.predict(x_val)
        y_pred_test = svm.predict(x_test)

        # Evaluation.
        train_acc, train_pr, train_rec, train_f1, train_auc = ml_utils.classification_metrics(y_train, y_pred_train)
        val_acc, val_pr, val_rec, val_f1, val_auc = ml_utils.classification_metrics(y_val, y_pred_val)
        test_acc, test_pr, test_rec, test_f1, test_auc = ml_utils.classification_metrics(y_test, y_pred_test)

        end = time()

        logger.info(f'Completed!\nTime Elapsed : {end - start:5.3f} seconds.')

        # Appending to record.
        record[ii] = (
                    train_acc, train_pr, train_rec, train_f1, train_auc,
                    val_acc, val_pr, val_rec, val_f1, val_auc,
                    test_acc, test_pr, test_rec, test_f1, test_auc
                    )
                    
        # Verbose.
        if args.svm.verbose:
            print(f'\nResults for k = {k}:')
            print('-'*50 + '\n')
            ml_utils.verbosity(train_acc, train_pr, train_rec, train_f1, train_auc, mode = 'train')
            ml_utils.verbosity(val_acc, val_pr, val_rec, val_f1, val_auc, mode = 'val')
            ml_utils.verbosity(test_acc, test_pr, test_rec, test_f1, test_auc, mode = 'test')

    record_df = pd.DataFrame(record, index = np.arange(1, k + 1), columns = columns)

    # Average values.
    logger.info(f'\nAverage Values:\n{record_df.mean(axis = 0).round(3)}\n')

    #%% To csv.
    cache_path = join(args.cache_dir, 'svm')
    if not exists(cache_path):
        os.mkdir(cache_path)
        
    record_df.to_csv(join(cache_path, f'svm_{str(args.svm.kernel)}_{str(args.svm.C)}_{str(args.svm.max_features)}_results.csv'))
    logger.info("Results Summary: \n")
    logger.info(f"{record_df.round(3)}")
