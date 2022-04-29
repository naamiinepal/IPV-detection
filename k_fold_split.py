# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 07:44:32 2022

@author: Sagun Shakya

Description:
    
    For Aspect Term Extraction Task:
        Takes in raw data for aspect term extraction, which is essentially a sequence tagging task and 
        writes the 5 - fold splits into text files named 'train.txt' and 'val.txt'.

    For Polarity Classification Task:
        Applies Stratified K-fold splits to the dataset (concat (IPV, non-IPV)) and 
        saves them in a TXT file (comma delimited).
"""
# Importing necessary libraries.
import argparse
import os

# Local Modules.
from utilities import utils
from utilities.kfold_split_asp_extraction_polarity import kfold_split_aspect, kfold_split_polarity

def argument_parser():
    parser = argparse.ArgumentParser(description = "Argument Parser for Creating K fold splits for aspect term extraction and Polarity Classification.")
    parser.add_argument('-t', '--type', default = 'aspect', choices = ['aspect', 'polarity'], type = str,
                        help = 'On which type of type to perform split.')
    parser.add_argument('-i', '--input_dir', default = r'data/aspect_extraction', type = str, metavar = 'PATH',
                        help = 'Path to raw data directory.')
    parser.add_argument('-o', '--save_dir', default = r'data/aspect_extraction/kfold', type = str, metavar ='PATH',
                        help = 'Path to the data directory to store K fold splits.')
    parser.add_argument('--log_dir', default = r'logs/splits', type = str, metavar = 'PATH',
                        help = 'Directory to save logs.')
    parser.add_argument('-k', '--kfolds', default = 5, type = int, 
                        help = 'Number of folds to generate.')                    
    parser.add_argument('-r', '--random_seed', default = 1234, type = int, 
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

    name_ = f"{args.type}_{args.kfolds}_fold_split.log"
    log_file = os.path.join(args.log_dir, name_)

    # Intialize Logger.
    logger = utils.get_logger(log_file)
    return logger
    
if __name__ == "__main__":
    args = argument_parser()
    logger = log_object(args)

    # Perform splits for either aspect term extraction task or Polarity classification task.
    if args.type == 'aspect':
        kfold_split_aspect(args, logger)
    else:
        kfold_split_polarity(args, logger)