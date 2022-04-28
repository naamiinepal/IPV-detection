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
from read_configuration import DotDict


def argument_parser():
    parser = argparse.ArgumentParser(description = "Argument Parser for Creating K fold splits fr aspect term extraction.")
    parser.add_argument('-i', '--input_dir', default = r'data/aspect_extraction', type = str, metavar = 'PATH',
                        help = 'Path to raw data directory.')
    parser.add_argument('-o', '--save_dir', default = r'data/aspect_extraction/kfold', type = str, metavar='PATH',
                        help = 'Path to the data directory to store K fold splits.')
    parser.add_argument('--log_dir', default = r'logs/splits', type = str, metavar = 'PATH',
                        help = 'Directory to save logs.')
    parser.add_argument('-k', '--kfolds', default=5, type=int, 
                        help = 'Number of folds to generate.')                    
    parser.add_argument('-r', '--random_seed', default=1234, type=int, 
                        help = 'Random seed.')
    parser.add_argument('-v', '--verbose', action = 'store_true', 
                        help = 'Whether to display verbose.')
    args = parser.parse_args()
    return args

def log_object(args): 
    '''
    Generates a logger object.

    Parameters
    ----------
    args : DotDict object.
        Arguments for the project.

    Returns
    -------
    logger : logger object on which log information can be written.
    '''      
    
    # If the log directory does not exist, we'll create it.
    os.makedirs(args.log_dir, exist_ok = True)

    name_ = f'asp_extraction_{args.kfolds}_fold_split.log'
    log_file = os.path.join(args.log_dir, name_)

    # Intialize Logger.
    logger = utils.get_logger(log_file)
    return logger

def main(args, logger):
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






