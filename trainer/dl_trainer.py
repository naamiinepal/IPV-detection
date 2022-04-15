# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:19:17 2022

@author: Sagun Shakya

Description:
    Main Trainer class.
"""


from transformers import AdamW
import wandb
from os import path, mkdir, makedirs
from pandas import DataFrame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(166)

from .evaluator import Evaluator
from utilities.utils import compute_prec_rec_f1

from tqdm import tqdm
tqdm.pandas(desc='Progress')

# Decay functions to be used with lr_scheduler
def lr_decay_noam(config):
    return lambda t: (
        10.0 * config.rnn.hidden_dim**-0.5 * min(
        (t + 1) * config.learning_rate_warmup_steps**-1.5, (t + 1)**-0.5))

def lr_decay_exp(config):
    return lambda t: config.learning_rate_falloff ** t


# Map names to lr decay functions
lr_decay_map = {
    'noam': lr_decay_noam,
    'exp': lr_decay_exp
}

# Trainer class
class Trainer():
    def __init__(self, config, logger, dataloader, model, device, k):
        """
            Trainer class. 
        """        
        self.config = config
        self.logger = logger
        self.dataloader = dataloader
        self.verbose = self.config.verbose

        self.train_dl, self.val_dl = dataloader.load_data(batch_size=config.batch_size)

        ### DO NOT DELETE
        ### DEBUGGING PURPOSE
#         sample = next(iter(self.train_dl))
#         print(sample.TEXT)
#         print(sample.LABEL)
#         print(sample.POS)
        
        self.train_dlen = len(self.train_dl)
        self.val_dlen = len(self.val_dl)
        #self.test_dlen = len(self.test_dl)
        
        self.epochs = self.config.epochs

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

        # To device.
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        # Optimizer.
        if self.config.model in ['mbert', 'muril']:
            self.opt = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                            lr = float(self.config.learning_rate), 
                            weight_decay = self.config.weight_decay)
        else:
            self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                    lr = float(self.config.learning_rate), 
                                    weight_decay=self.config.weight_decay)
        
        self.lr_scheduler_step = self.lr_scheduler_epoch = None
        
        # Set up learing rate decay scheme.
        if self.config.use_lr_decay:
            if '_' not in self.config.lr_rate_decay:
                raise ValueError("Malformed learning_rate_decay")
            lrd_scheme, lrd_range = self.config.lr_rate_decay.split('_')

            if lrd_scheme not in lr_decay_map:
                raise ValueError("Unknown lr decay scheme {}".format(lrd_scheme))
            
            lrd_func = lr_decay_map[lrd_scheme]            
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                                            self.opt, 
                                            lrd_func(self.config),
                                            last_epoch=-1
                                        )
            # For each scheme, decay can happen every step or every epoch
            if lrd_range == 'epoch':
                self.lr_scheduler_epoch = lr_scheduler
            elif lrd_range == 'step':
                self.lr_scheduler_step = lr_scheduler
            else:
                raise ValueError("Unknown lr decay range {}".format(lrd_range))

        self.k = k
        self.model_name = f'{self.config.model}_{self.config.train_type}_{self.config.learning_rate}_Fold_{self.k}'
        self.file_name = self.model_name + '.pth'
        makedirs(path.join(self.config.output_dir, self.config.model), exist_ok = True)
        self.model_file = path.join(self.config.output_dir, self.config.model, self.file_name)
        
        self.total_train_loss = []
        self.total_train_acc = []
        self.total_val_loss = []
        self.total_val_acc = []
        
        self.early_max_patience = self.config.early_max_patience
        
    # Load saved model
    def load_checkpoint(self):
        """
        Loads the trained model.
        """   
        
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.opt = checkpoint['opt']
        self.opt.load_state_dict(checkpoint['opt_state'])
        self.total_train_loss = checkpoint['train_loss']
        #self.total_train_acc = checkpoint['train_acc']
        self.total_val_loss = checkpoint['val_loss']
        #self.total_val_acc = checkpoint['val_acc']
        self.epochs = checkpoint['epochs']
        
        
    # Save model.
    def save_checkpoint(self):
        """
        Saves the trained model.
        """      
        save_parameters = {'state_dict': self.model.state_dict(),
                           'opt': self.opt,
                           'opt_state': self.opt.state_dict(),
                           'train_loss' : self.total_train_loss,
                           'val_loss' : self.total_val_loss,
                           'epochs' : self.epochs}
        torch.save(save_parameters, self.model_file)        

    # Get the accuracy per batch
    def categorical_accuracy(self, preds, y):
        '''
        Calculates the accuracy for the given batch.

        Parameters
        ----------
        preds : int
             predicted labels.
        y : int
            gold labels.

        Returns
        -------
        float
            Batch-wise accuracy.

        '''
        
        # Get the index of the max probability.
        max_preds = preds.argmax(dim = 1, keepdim = True).squeeze(1)    # Shape -> (batch_size)
        correct = max_preds.eq(y)
        return correct.sum().item() / torch.FloatTensor([y.shape[0]])

    
    def train(self, model, iterator, optimizer, criterion):
        '''
        Trains the given model.

        Parameters
        ----------
        model : object.
            Pytorch model. Can be one of {lstm, cnn}
        iterator : iter
            dataset iterator.
        optimizer : 
            Optimizer for backpropagation.
        criterion : 
            loss function.

        Returns
        -------
        Average train loss and accuracy for the current epoch.

        '''
        epoch_loss = 0
        epoch_acc = 0
        
        # The lengths of gold and pred list should be equal to the total number of sentences encountered.
        gold_list = []
        pred_list = []
        
        model.train()

        for coll in iterator:
            X = coll.TEXT
            Y = coll.IPV

            # To device.
            if self.config.model in ['mbert', 'muril']:
                # Create an attention mask.
                mask = (X > 0).to(int)
                mask = mask.to(self.device)
                X = X.to(self.device)
            else:
                X = X.to(self.device)
                Y = Y.to(self.device)

            optimizer.zero_grad()
                        
            predictions = model(X, mask) if self.config.model in ['mbert', 'muril'] else model(X)         # Shape -> (batch_size, 2)
            gold = Y

            gold = gold.squeeze(1) if len(gold.shape) > 1 else gold                    # Shape -> (batch_size)

            loss = criterion(predictions, gold)
            
            # Normalize the raw logits using softmax.
            predictions = F.softmax(predictions, dim = -1)                          # Shape -> (batch_size, 2)
            
            # Calculate accuracy.
            acc = self.categorical_accuracy(predictions, gold)
            
            y_pred = predictions.argmax(dim = 1, keepdim = True).squeeze(1)         # Shape -> (batch_size)
            
            # Appending to the gold list and pred list.
            gold_list += gold.tolist()
            pred_list += y_pred.tolist()
            
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        # Average epoch loss and accuracy.
        avg_epoch_loss = epoch_loss / self.train_dlen
        avg_epoch_acc = epoch_acc / self.train_dlen 
        
        assert len(gold_list) == len(pred_list), f"The gold list has length {len(gold_list)} while the predicted list has length {len(pred_list)}."
        
        return avg_epoch_loss, avg_epoch_acc, gold_list, pred_list   

    def evaluate(self, model, iterator, criterion):
        '''
        Evaluates the model for the validation set.

        Parameters
        ----------
        model : torch model
        iterator : valid_dataloader.
        criterion : loss function.

        Returns
        -------
        avg_epoch_loss, avg_epoch_acc -- average loss and accuracy for the current epoch.
        gold_list, pred_list -- gold labels and predicted labels.
        '''
        epoch_loss = 0
        epoch_acc = 0

        model.eval()
        
        gold_label = []
        pred_label = []
        
        with torch.no_grad():

            for coll in iterator:
                X = coll.TEXT
                Y = coll.IPV

                # To device.
                if self.config.model in ['mbert', 'muril']:
                    # Create an attention mask.
                    mask = (X > 0).to(int)
                    mask = mask.to(self.device)
                    X = X.to(self.device)
                else:
                    X = X.to(self.device)
                    Y = Y.to(self.device)

                predictions = model(X, mask) if self.config.model in ['mbert', 'muril'] else model(X)          # Shape -> (batch_size, 2)
                gold = Y                         # Shape -> (batch_size, 1)
                
                # True label.
                gold = gold.squeeze(1) if len(gold.shape) > 1 else gold            # Shape -> (batch_size)                        
                gold_label.append(gold.data.cpu().numpy().tolist())

                # Calculate loss.
                loss = criterion(predictions, gold)
                
                # Predicted Label.
                predictions = F.softmax(predictions, dim = -1)          # Shape -> (batch_size, 2)
                acc = self.categorical_accuracy(predictions, gold)      

                # Calculate accuracy.
                pred = predictions.argmax(dim = 1, keepdim = True).squeeze(1).data.cpu().numpy().tolist()  # Shape -> (batch_size)
                pred_label.append(pred)

                # Compute loss and accuracy.
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
                
        gold_list = [y for x in gold_label for y in x]
        pred_list = [y for x in pred_label for y in x]
        
        assert len(gold_list) == len(pred_list), f"The gold list has length {len(gold_list)} while the predicted list has length {len(pred_list)}."
        
        avg_epoch_loss =  epoch_loss / self.val_dlen
        avg_epoch_acc = epoch_acc / self.val_dlen
        
        
        return avg_epoch_loss, avg_epoch_acc, gold_list, pred_list

    
    
    def fit(self):
        """
        Trains and evaluates the given model.
        """   
        
        # Store info regarding loss and other metrics.
        cache = {'training loss': [],
                 'training accuracy': [],
                 'training precision': [],
                 'training recall': [],
                 'training f1 score': [],
                 'training roc-auc score': [],
                 'validation loss': [],
                 'validation accuracy': [],
                 'validation precision': [],
                 'validation recall': [],
                 'validation f1 score': [],
                 'validation roc-auc score': []}
        
        best_valid_loss = float('inf')
        best_valid_acc = 0.0
        valid_cache = []
        counter = 0

        for epoch in range(0, self.epochs):

            tqdm_t = tqdm(iter(self.train_dl), leave=False, total=self.train_dlen)
            tqdm_v = tqdm(iter(self.val_dl), leave=False, total=self.val_dlen)
            
            train_loss, train_acc, train_gold_list, train_pred_list = self.train(self.model, tqdm_t, self.opt, self.loss_fn)
            valid_loss, valid_acc, valid_gold_list, valid_pred_list = self.evaluate(self.model, tqdm_v, self.loss_fn)
            
            # Precision, Recall, F1 Score and ROC-AUC Score.
            train_pr, train_rec, train_f1, train_auc = compute_prec_rec_f1(train_gold_list, train_pred_list, self.config.average)
            valid_pr, valid_rec, valid_f1, valid_auc = compute_prec_rec_f1(valid_gold_list, valid_pred_list, self.config.average)
            
            # Store the logs in a dictionary.
            log_dict = {'training loss': train_loss,
                        'training accuracy': train_acc,
                        'training precision': train_pr,
                        'training recall': train_rec,
                        'training f1 score': train_f1,
                        'training roc-auc score': train_auc,
                        'validation loss': valid_loss,
                        'validation accuracy': valid_acc,
                        'validation precision': valid_pr,
                        'validation recall': valid_rec,
                        'validation f1 score': valid_f1,
                        'validation roc-auc score': valid_auc
                        }
            
            # Store in cache.
            for key in log_dict:
                cache[key] += [log_dict[key]]
                
            # wandb.
            # Record train and val loss and accuracy.
            if self.config.wandb_config.WandB:
                wandb.log(log_dict)
            
            # Save checkpoint in case of Best validation loss.
            if valid_loss < best_valid_loss:
                self.save_checkpoint()
                best_valid_loss = valid_loss
                counter=0
                if self.verbose:
                    self.logger.info(f"Best model saved at {self.model_file}.\n")

                # Save cache.
                valid_cache = [valid_loss, valid_acc, valid_pr, valid_rec, valid_f1, valid_auc]

            else:
                counter += 1
                if self.verbose:
                    self.logger.info(f'No improvement in validation loss. Tolerance count : {counter}\n')            
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
            
            # Verbosity.
            self.logger.info(f'Epoch: {epoch+1}\n')
            
            # Train Verbose.
            train_verbose1 = f'Train Loss: {train_loss:.3f} || Train Acc: {train_acc:.3f}\n'
            train_verbose2 = f'Train Precision: {train_pr:.3f} || Train Recall: {train_rec:.3f} || Train F1 Score:{train_f1:.3f} || Train ROC-AUC Score: {train_auc:.3f}\n'
            
            if self.verbose:
                self.logger.info(train_verbose1 + train_verbose2)
            
            # Valid verbose.
            valid_verbose1 = f'\nValid Loss: {valid_loss:.3f} || Valid Acc: {valid_acc:.3f}\n'
            valid_verbose2 = f'Valid Precision: {valid_pr:.3f} || Valid Recall: {valid_rec:.3f} || Valid F1 Score:{valid_f1:.3f} || Valid ROC-AUC Score: {valid_auc:.3f}\n'
            
            if self.verbose:
                self.logger.info(valid_verbose1 + valid_verbose2)
            
            # Check for early stopping.
            if counter >= self.early_max_patience: 
                if self.verbose:
                    self.logger.info(f"Training stopped because maximum tolerance of {self.early_max_patience} reached.")
                break
        
        if self.config.wandb_config.WandB:
            wandb.summary['best validation loss'] = best_valid_loss
            wandb.summary['best validation accuracy'] = best_valid_acc
        
        cache_dir = self.config.cache_dir
        if not path.exists(cache_dir):
            mkdir('./cache_dir')
            cache_dir = './cache_dir'

        cache_dir = path.join(cache_dir, self.config.model)
        if not path.exists(cache_dir):
            mkdir(cache_dir)

        # Output file path.
        cache_filename = path.join(cache_dir, f'cache_{self.config.model}_{self.config.train_type}_Fold_{str(self.k)}.csv')
        
        # Pandas DataFrame.
        cache_df = DataFrame(cache)
        
        # Convert to csv file.
        cache_df.to_csv(cache_filename, index = None, sep = ',')

        return valid_cache
    
    # Predict
    def predict(self):
        '''
        Evaluates on test set after each run.

        Returns
        -------
        test_acc : float
            Accuracy.
        prec : float
            Precision.
        rec : float
            Recall.
        f1 : float
            F1 Score.
        auc : float
            ROC-AUC Score.

        '''        
        evaluate = Evaluator(self.config, self.logger, self.model, self.dataloader, self.model_name)
        
        self.model.eval()
        tqdm_tst = tqdm(iter(self.test_dl), leave = False, total = self.test_dlen)      
        test_loss, test_acc, gold_list, pred_list = self.evaluate(self.model, tqdm_tst, self.loss_fn)
        self.logger.info(f'Test. Loss: {test_loss:.3f} || Test Acc: {test_acc*100:.4f}%')
        
        # Precision, recall and f1 score + ROC-AUC Score.
        prec, rec, f1, auc = evaluate.prec_rec_f1(gold_list, pred_list)
        
        
        self.logger.info("Writing results")
        evaluate.write_results()
        
        return (test_acc, prec, rec, f1, auc)
    
    
    def infer(self, sent, aspect_term, aspect_cat):
        """
        Returns inference on a given input tuple.
        """
        evaluate = Evaluator(self.config, self.logger, self.model, self.dataloader, self.model_name)
        return evaluate.infer(sent, aspect_term, aspect_cat)

