# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:44:18 2022

@author: Sagun Shakya
"""
import os
import argparse

from models.ml_models import svm
from utilities import utils
from utilities.read_configuration import DotDict

def parse_args_yaml(config_file = 'config.yaml'):
    '''
    Parses the arguments from config file (in YAML format).

    Parameters
    ----------
    config_file : str, optional
        Path to the config file. The default is 'config.yaml'.

    Returns
    -------
    args : DotDict object
        Arguments to be used in the project. The values can be accessed by using notation.

    '''
    # Processing the hyperparameters.
    configuration = utils.load_config(config_file)
    
    return configuration

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

    if args.svm:
        name_ = f'svm_log_{args.svm.kernel}_{str(args.svm.C)}_{utils.timestamp()}.log'
        log_file = os.path.join(args.log_dir, name_)

    else:    
        # Name output log file.
        log_suffix = '_' + args.model + '_' + str(args.train_type) + '.log'
        log_file = os.path.join(args.log_dir, 'nepsa' + log_suffix)
        
    # Intialize Logger.
    logger = utils.get_logger(log_file)
    return logger

def parse_arguments():
    '''
    Argument parser for IPV project.

    Returns
    -------
    args : TYPE
        Arguments for the run.

    '''
    parser = argparse.ArgumentParser(description="Online IPVDetection argument parser.")
    parser.add_argument('--svm', action = 'store_true', 
                        help = "Whether to perform SVM.")
    parser.add_argument('--nb', action = 'store_true', 
                        help = "Whether to perform Naive Bayes.")
    args = parser.parse_args()
    
    return args
    
def main():
    '''
    Main function that is called in the driver code.
    '''
    # From CLI.
    arguments = parse_arguments()
    
    # From YAML.
    config = parse_args_yaml(config_file = 'config.yaml')
    
    # Append to a single dictionary.
    configuration = dict(vars(arguments), **config)
    
    # Access the configuration values using dot notation.
    args = DotDict(configuration)
    
    # Instantiate logger object.
    logger = log_object(args)
    
    if args.svm:
        svm.main(args, logger)
    
    


if __name__=='__main__':
    '''
    Driver Code.
    '''
    main()
