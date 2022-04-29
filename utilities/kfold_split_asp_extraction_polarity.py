# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 07:44:32 2022

@author: Sagun Shakya

Description:
    Applies K-fold splits to the dataset asp_extraction.tsv and 
    saves them in a TXT file (comma delimited) for each fold.
"""

from collections import Counter
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
    '''
    Takes in raw data for aspect term extraction, which is essentially a sequence tagging task and 
    writes the 5 - fold splits into text files named 'train.txt' and 'val.txt'.
    '''
    # Read raw data.
    data = pd.read_csv(join(args.input_dir, 'asp_extraction.tsv'), 
                            header = None,
                            delimiter = '\t', 
                            encoding = 'utf-8')

    data.columns = "tokens labels".split()

    # Drop duplicates.
    shape_before = data.shape
    data.drop_duplicates(subset=['tokens'], keep = 'first', inplace = True)
    shape_after = data.shape

    # Verbosity.
    if args.verbose:
        logger.info('Removing duplicate rows...\n')
        logger.info(f'Shape before : {shape_before}.')
        logger.info(f'Shape after: {shape_after}.')
        logger.info(f'Number of duplicates removed : {shape_before[0] - shape_after[0]}.\n')
    
    # Basic pre-processing.
    # Number of duplicate rows: 3488 - 314 = 3174.

    # Convert the string objects to list.
    for col in data.columns:
        data[col] = data[col].apply(literal_eval)

    # Removing sentence having only "O" tokens: 3174 - 2538 = 636.
    shape_before = data.shape

    data['labels'] = data['labels'].apply(lambda label: np.nan if all(element=="O" for element in label) else label)
    data.dropna(inplace = True)
    data.reset_index(drop = True, inplace = True)

    shape_after = data.shape

    # Verbosity.
    if args.verbose:
        logger.info('Removing "O" tokens')
        logger.info(f'\nShape before : {shape_before}.')
        logger.info(f'Shape after: {shape_after}.')
        logger.info(f'Number of sentences with only "O" tags removed : {shape_before[0] - shape_after[0]}.\n')

    # K - fold splits.
    kf = KFold(n_splits = args.kfolds, random_state = args.random_seed, shuffle = True)

    for k, (train_id, val_id) in enumerate(kf.split(data['tokens'], data['labels']), 1):
        train = data.loc[train_id, :]
        val = data.loc[val_id, :]

        # Pre-processing.

        # Info.
        logger.info(f'Fold : {k}\n' + f'{"-" * 50}')
        logger.info(f'\nLength of training set : {len(train)}')
        logger.info(f'Length of validation set : {len(val)}\n')

        # Save files to CONLL2003 format.
        save_filepath = join(args.save_dir, str(k))
        os.makedirs(save_filepath, exist_ok=True)
        
        if args.verbose:
            logger.info(f'Saving files at : {save_filepath}...\n')

        # Write the data to CoNLL format.
        utils.write_to_CONLL(train, join(save_filepath, 'train.txt'))
        utils.write_to_CONLL(val, join(save_filepath, 'val.txt'))

        # Value counter.
        if args.verbose:
            labels = val['labels'].sum()
            asp_counts = Counter(labels)
            logger.info("Aspect Categories Frequency Distribution:\n", asp_counts)
        
        if args.verbose:    
            logger.info(f"\nSuccess! Save date : {utils.current_timestamp()}\n")

def kfold_split_polarity(args, logger):
    #%% Read raw data.

    # Load IPV examples.
    ipv = pd.read_csv(join(args.input_dir, 'IPV_sents.tsv'), delimiter = '\t', encoding = 'utf-8')
    ipv.columns = "id sents".split()
    ipv['pol'] = np.ones(len(ipv), dtype = np.int8)

    # Load IPV examples.
    non_ipv = pd.read_csv(join(args.input_dir, 'non-IPV_sents.tsv'), delimiter = '\t', encoding = 'utf-8')
    non_ipv.columns = "id sents".split()
    non_ipv['pol'] = np.zeros(len(non_ipv), dtype = np.int8)

    #%% Verbose.
    logger.info("Number of IPV instances : {}".format(len(ipv)))
    logger.info("Number of non-IPV instances : {}".format(len(non_ipv)))

    #%% Concat both and shuffle.
    df = pd.concat([ipv, non_ipv], axis = 0).sample(frac = 1, random_state = args.random_seed)
    df.reset_index(drop = True, inplace = True)

    # Stratified k fold.
    skf = StratifiedKFold(n_splits = args.kfolds)
    target = df['pol'].values

    for k, (train_id, val_id) in enumerate(skf.split(df, target), 1):
        train = df.loc[train_id, :]
        val = df.loc[val_id, :]

        # Info.
        logger.info(f'\nLength of training set : {len(train)}')
        logger.info(f'Length of validation set : {len(val)}\n')

        # Save files.
        save_filepath = join(args.save_dir, str(k))
        os.makedirs(save_filepath, exist_ok=True)
        
        logger.info(f'Saving files at : {save_filepath}...\n')

        utils.write_csv(train, join(save_filepath, 'train.txt'))
        utils.write_csv(val, join(save_filepath, 'val.txt'))
        
        logger.info(f"Success! Save date : {utils.current_timestamp()}\n")

    #%% Verbose.
    if args.verbose:
        train_coll = {'0' : [], '1' : []}
        val_coll = {'0' : [], '1' : []}

        for k in range(1, 6):
            save_filepath = join(args.save_dir, str(k))
            train_df = pd.read_csv(join(save_filepath, 'train.txt'), header = None, encoding = 'utf-8')
            val_df = pd.read_csv(join(save_filepath, 'val.txt'), header = None, encoding = 'utf-8')
            
            train_dict = train_df.iloc[:, -1].value_counts().to_dict()
            val_dict = val_df.iloc[:, -1].value_counts().to_dict()
            
            train_coll['0'].append(train_dict[0])
            train_coll['1'].append(train_dict[1])

            val_coll['0'].append(val_dict[0])
            val_coll['1'].append(val_dict[1])

        logger.info("Size of train data across 5 folds : \n", train_coll)
        logger.info("Size of validation data across 5 folds : \n", val_coll)




