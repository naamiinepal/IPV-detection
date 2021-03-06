# -*- coding: utf-8 -*-
"""
Created on Wed Apr 6 20:15:06 2022

@author: Sagun Shakya

Dataloader for CSV file for a particular fold k_i.
"""

import os
import torch
from torchtext.legacy import data
from torchtext import vocab
from torchtext import datasets
from transformers import BertTokenizer

# Local modules.
from utilities.read_configuration import DotDict


class Dataloader():

    def __init__(self, args: DotDict, k: int, device: str):
        '''
        Dataloader class to generate the data iterators for the training, validation and testing set.

        Parameters
        ----------
        args : DotDict instance
            From configuration file.
        k : int
            Fold number in the current run.
            This will get increased progressively on each run.
        device : str
            Which device to use out of {'cpu', 'cuda'}.
        '''
        self.device = device
        self.data_path = os.path.join(args.data_path, str(k))
        self.batch_size = args.batch_size
        
        if args.model == 'bert' or args.model == 'muril':
            self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert.model_name) 
            self.PAD_INDEX = self.bert_tokenizer.pad_token
            self.CLS_INDEX = self.bert_tokenizer.cls_token
            self.UNK_INDEX = self.bert_tokenizer.unk_token

            # Tokenizer function that has input_ids and attention mask.
            if args.train_type == 'text':
                self.tokenizer_funct = lambda text: self.bert_tokenizer(text, padding="max_length", max_length = 512, truncation=True, return_tensors = "pt")
            elif args.train_type == 'atsa':
                self.tokenizer_funct = lambda text, at: self.bert_tokenizer(text, at, padding="max_length", max_length = 512, truncation=True, return_tensors = "pt")
            elif args.train_type == 'acsa':
                self.tokenizer_funct = lambda text, ac: self.bert_tokenizer(text, ac, padding="max_length", max_length = 512, truncation=True, return_tensors = "pt")

            self.txt_field = data.Field(sequential = True,
                                        tokenize = self.bert_tokenizer.tokenize,
                                        batch_first = True,
                                        pad_token = self.PAD_INDEX,
                                        init_token = self.CLS_INDEX,
                                        unk_token = self.UNK_INDEX)
        else:
            self.txt_field = data.Field(tokenize = self.tokenizer, use_vocab = True, unk_token = '<unk>', batch_first = True)

        self.ipv_field = data.Field(batch_first=True, 
                                    is_target = True,
                                    unk_token=None, 
                                    pad_token=None, 
                                    use_vocab = False, 
                                    sequential = False 
                                    )

        #self.id_field = data.Field(unk_token='<unk>', batch_first=True)
        #self.at_field = data.Field(tokenize=self.tokenizer, use_vocab=True, unk_token='<unk>', batch_first=True)
        #self.ac_field = data.Field(batch_first=True, unk_token=None, pad_token=None)
          
        #self.fields = (('IPV', self.ipv_field), ('ASPECT', self.ac_field), 
        #               ('TERM', self.at_field), ('TEXT', self.txt_field))            

        self.fields = ((None, None), ('TEXT', self.txt_field), ('IPV', self.ipv_field))          # THIS WILL BE MODIFIED AFTER ASPECT TERMS AND CATEGORIES ARE ADDED.

        self.train_ds, self.val_ds = data.TabularDataset.splits(path=self.data_path, 
                                                                format='csv', 
                                                                train='train.txt', 
                                                                validation='val.txt',                         
                                                                fields=self.fields)

        self.embedding_dir = args.emb_dir
        self.vec = vocab.Vectors(name=args.emb_file, cache=self.embedding_dir)

        if args.model == 'bert' or args.model == 'muril':
            self.txt_field.build_vocab(self.train_ds.TEXT)
            self.txt_field.vocab.stoi = self.bert_tokenizer.vocab
            self.txt_field.vocab.itos = list(self.bert_tokenizer.vocab)
            
        else:
            self.txt_field.build_vocab(self.train_ds.TEXT, self.val_ds.TEXT, max_size=None, vectors=self.vec)
        
        #self.at_field.build_vocab(self.train_ds.TERM, self.val_ds.TERM, max_size=None, vectors=self.vec)
        #self.ac_field.build_vocab(self.train_ds.ASPECT)
        #self.id_field.build_vocab(self.train_ds.ID)
        self.ipv_field.build_vocab(self.train_ds.IPV)
        
        self.vocab_size = len(self.txt_field.vocab)
        #self.at_size = len(self.at_field.vocab)
        #self.ac_size = len(self.ac_field.vocab)
        self.ipv_size = len(self.ipv_field.vocab)
        
        self.tagset_size = self.ipv_size            # CHANGE THIS FOR ASPECT TERM EXTRACTION.
        
        self.weights = self.txt_field.vocab.vectors

        self.print_stat()

    def tokenizer(self, x):
        return x.split()        
    
    def train_ds(self):
        return self.train_ds
    
    def val_ds(self):
        return self.val_ds    

    def train_ds_size(self):
        return len(self.train_ds)
    
    def val_ds_size(self):
        return len(self.val_ds)  
    
    def txt_field(self):
        return self.txt_field
    
    def ipv_field(self):
        return self.ipv_field    
    
    def vocab_size(self):
        return self.vocab_size

    def tagset_size(self):
        return self.tagset_size

    #def at_size(self):
    #    return self.at_size
    
    #def ac_size(self):
    #    return self.ac_size
    
    def weights(self):
        return self.weights
    
    def print_stat(self):
        """
        Prints the data statistics.
        """
        print('\nLocation of dataset = ', self.data_path)
        print('Length of training dataset = ', len(self.train_ds))
        print('Length of validation dataset = ', len(self.val_ds))
        print('Length of text vocab (unique words in dataset) = ', self.vocab_size)
        print('Length of label vocab (unique tags in labels) = ', self.tagset_size)
        print()
    
    def load_data(self, batch_size: int, shuffle: bool = False):
        '''
        Generates the data iterators for train, validation and test data.

        Parameters
        ----------
        batch_size : int
            batch_size.
        shuffle : Bool, optional
            Whether to shuffle the data before training/testing. The default is True.

        Returns
        -------
        train_iter : training Dataloader instance.
        val_iter : validation Dataloader instance.
        '''
        
        train_iter, val_iter = data.BucketIterator.splits(datasets=(self.train_ds, self.val_ds), 
                                                        batch_sizes=(batch_size, batch_size), 
                                                        sort_key=lambda x: len(x.TEXT), 
                                                        device=self.device, 
                                                        sort_within_batch=True, 
                                                        repeat=False,
                                                        shuffle=shuffle)

        return train_iter, val_iter
