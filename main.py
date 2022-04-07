# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:44:18 2022

@author: Sagun Shakya
"""
import os
import argparse

from trainer.ml_trainer import MLTrainer
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
    
    name_ = f'{args.model}_{args.vectorizer.mode}_{args.vectorizer.max_features}.log'
    log_file = os.path.join(args.log_dir, name_)

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
    CHOICES = ['svm', 'nb', 'random_forest', 'adaboost', 'logistic_regression'] + ['lstm', 'gru', 'cnn', 'mbert']
    parser = argparse.ArgumentParser(description="Online IPV Detection argument parser.")
    parser.add_argument('-m', '--model', choices=CHOICES,  default = 'adaboost',
                        help = 'Type of model to run.')
    parser.add_argument('-t', '--train_type', choices = ['text', 'atsa', 'acsa', 'concat'], default = 'text',
                        help = 'Type of training: Should be one of {"text : text only", "atsa : text + at", "acsa : text + ac", "concat : text + at + ac"}.')
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
    
    # Train ML Classifier.
    ml_trainer = MLTrainer(args, logger)
    ml_trainer.train()


if __name__=='__main__':
    '''
    Driver Code.
    '''
    main()
