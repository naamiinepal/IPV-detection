# -*- coding: utf-8 -*-
"""
Created on Mon Apr  26 09:44:18 2022

@author: Sagun Shakya
"""
import argparse
import os
import random

import torch
from numpy import empty
from numpy import random as rdm
from pandas import DataFrame
from transformers import BertTokenizerFast

from dataloader.asp_dataloader import AspectExtractionCorpus_Transformer

# Local Modules.
from dataloader.dl_dataloader import Dataloader
from trainer.dl_trainer import Trainer
from utilities import utils
from utilities.read_configuration import DotDict

# Determinism.
SEED = 1234
random.seed(SEED)
rdm.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizerFast.from_pretrained("google/muril-base-cased")
dataloader = AspectExtractionCorpus_Transformer(
    "D:\\ML_projects\\IPV-Project\\data\\aspect_extraction\\kfold\\2", tokenizer, 128
)

train_dl, val_dl = dataloader.load_data(batch_sizes=(8, 8), shuffle=True)

print(f"Train data : {next(iter(train_dl))}")
print(f"\nVal data : {next(iter(val_dl))}")
