# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 07:44:32 2022

@author: Sagun Shakya
"""
import os
from os.path import join
import pandas as pd
import numpy as np
import argparse
from parso import parse
from sklearn.model_selection import StratifiedKFold

# Local Modules.
from utilities import utils
from read_configuration import DotDict


def argument_parser():
    parser = argparse.ArgumentParser(description = "Argument Parser for Creating stratified K fold splits.")
    parser.add_argument('-i', '--input_dir', default = r'data/raw', type = str, metavar = 'PATH',
                        help = 'Path to raw data directory.')
    parser.add_argument('-o', '--save_dir', default = r'data/kfold', type = str, metavar='PATH',
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
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    name_ = f'Stratified_{args.kfolds}_fold_split.log'
    log_file = os.path.join(args.log_dir, name_)

    # Intialize Logger.
    logger = utils.get_logger(log_file)
    return logger

def main(args, logger):
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

if __name__ == '__main__':
    '''
    Driver Code.
    '''

    # Parse arguments.
    args = argument_parser()

    # Access the configuration values using dot notation.
    args = DotDict(vars(args))

    # Instantiate logger object.
    logger = log_object(args)

    # Perform K fold Split.
    main(args, logger)

