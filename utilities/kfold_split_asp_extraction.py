# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 07:44:32 2022

@author: Sagun Shakya

Description:
    Applies K-fold splits to the dataset (concat (IPV, non-IPV)) and 
    saves them in a TXT file (comma delimited).
"""

import os
from os.path import join
import pandas as pd
import numpy as np
import argparse
from ast import literal_eval
from sklearn.model_selection import StratifiedKFold, KFold

# Local Modules.
from utilities import utils


def kfold_split_aspect(args, logger):
    #%% Read raw data.

    root = r'D:\ML_projects\IPV-Project\data\aspect_extraction'
    # Load examples.
    data = pd.read_csv(join(root, 'asp_extraction.tsv'), 
                            header = None,
                            delimiter = '\t', 
                            encoding = 'utf-8')

    data.columns = "tokens labels".split()

    print('Removing duplicate rows...\n')
    shape_before = data.shape
    print(f'\nShape before : {shape_before}.')

    data.drop_duplicates(subset=['tokens'], keep = 'first', inplace = True)

    shape_after = data.shape
    print(f'Shape after: {shape_after}.')
    print(f'Number of duplicates removed : {shape_before[0] - shape_after[0]}.\n')
    
    # Basic pre-processing.
    # Number of duplicate rows: 3488 - 314 = 3174.
    for col in data.columns:
        data[col] = data[col].apply(literal_eval)

    # Removing "O" tokens: 3174 - 2538 = 636.
    print('Removing "O" tokens')
    shape_before = data.shape
    print(f'\nShape before : {shape_before}.')

    data['labels'] = data['labels'].apply(lambda label: np.nan if all(element=="O" for element in label) else label)
    data.dropna(inplace = True)
    data.reset_index(drop = True, inplace = True)

    shape_after = data.shape
    print(f'Shape after: {shape_after}.')
    print(f'Number of sentences with only "O" tags removed : {shape_before[0] - shape_after[0]}.\n')

    # K - fold splits.
    kf = KFold(n_splits = args.kfolds, random_state = args.random_seed, shuffle = True)

    for k, (train_id, val_id) in enumerate(kf.split(data['tokens'], data['labels']), 1):
        train = data.loc[train_id, :]
        val = data.loc[val_id, :]

        # Pre-processing.

        # Info.
        logger.info(f'\nLength of training set : {len(train)}')
        logger.info(f'Length of validation set : {len(val)}\n')

        # Save files to CONLL2003 format.
        save_filepath = join(args.save_dir, str(k))
        os.makedirs(save_filepath, exist_ok=True)
        
        logger.info(f'Saving files at : {save_filepath}...\n')

        utils.write_to_CONLL(train, join(save_filepath, 'train.txt'))
        utils.write_to_CONLL(val, join(save_filepath, 'val.txt'))
        
        logger.info(f"Success! Save date : {utils.current_timestamp()}\n")





