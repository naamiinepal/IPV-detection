# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:44:18 2022

@author: Sagun Shakya
"""
import os
import argparse
import random
from numpy import empty, random as rdm
from pandas import DataFrame
import torch

# Local Modules.
from dataloader.dl_dataloader import Dataloader
from models.dl_models.models import RNN
from trainer.ml_trainer import MLTrainer
from trainer.dl_trainer import Trainer
from utilities import utils
from utilities.read_configuration import DotDict

# Determinism.
SEED = 1234
random.seed(SEED)
rdm.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def parse_arguments():
    '''
    Argument parser for IPV project.

    Returns
    -------
    args : TYPE
        Arguments for the run.

    '''
    MODEL_CHOICES = ['svm', 'nb', 'random_forest', 'adaboost', 'logistic_regression'] + ['lstm', 'gru', 'cnn', 'mbert']
    parser = argparse.ArgumentParser(description="Online IPV Detection argument parser.")

    parser.add_argument('-m', '--model', choices = MODEL_CHOICES,  default = 'lstm',
                        help = 'Type of model to run.')
    parser.add_argument('-t', '--train_type', choices = ['text', 'atsa', 'acsa', 'concat'], default = 'text',
                        help = 'Type of training: Should be one of {"text : text only", "atsa : text + at", "acsa : text + ac", "concat : text + at + ac"}.')
    
    args = parser.parse_args()
    
    return args

def train_dl_model(args: DotDict, logger: utils.log_object, device: str):

    cache_df = DataFrame(empty((5, 6)), 
                        columns = ['validation loss',
                                    'validation accuracy',
                                    'validation precision',
                                    'validation recall',
                                    'validation f1 score',
                                    'validation roc-auc score']
                        )

    logger.info(f'Training Started on {utils.current_timestamp()}.')
    for k in [1]:
        data_loader = Dataloader(args, k, device)
        train_dl, val_dl = data_loader.load_data(args.batch_size)

        # Instantiate model.
        if args.model == 'lstm':
            model = RNN(args, data_loader)
        
        assert model is not None, "Instantiate model!"

        # Reset model weights to avoid weight leakage.
        utils.reset_weights(model)

        # Set up trainer.
        trainer = Trainer(args, logger, data_loader, model, k)
        cache = trainer.fit()
        cache_df.iloc[k - 1, :] = cache

    # Remove empty rows (if any).
    cache_df = cache_df[cache_df > 1e-5].dropna()

    # Take average across all folds.
    cache_df = cache_df.append([cache_df.mean(), cache_df.std()], ignore_index = True)

    # Save cache for each fold.
    cache_save_dir = os.path.join(args.cache_dir, args.model)

    # Make directory if it doesn't exist.
    os.makedirs(cache_save_dir, exist_ok=True)
    cache_filename = f'Val_results_{args.model}_{args.train_type}_{args.learning_rate}_{utils.timestamp()}.csv'
    cache_df.to_csv(os.path.join(cache_save_dir, cache_filename), index = None)

    logger.info(f'Training Completed at {utils.current_timestamp()}!\n')
    logger.info(f'Results of the best model saved at : {cache_save_dir}')


def main():
    '''
    Main function called in the driver code.
    '''

    # From CLI.
    arguments = parse_arguments()
    
    # From YAML.
    config = utils.parse_args_yaml(config_file = 'config.yaml')
    
    # Append to a single dictionary.
    configuration = dict(vars(arguments), **config)
    
    # Access the configuration values using dot notation.
    args = DotDict(configuration)
    
    # Instantiate logger object.
    logger = utils.log_object(args)
    
    # Whether to use CUDA.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f'\nRunning on CUDA : {use_cuda}\n')   

    # Train ML Classifier.
    #ml_trainer = MLTrainer(args, logger)
    #ml_trainer.train()

    train_dl_model(args, logger, device)


if __name__=='__main__':
    '''
    Driver Code.
    '''
    main()
