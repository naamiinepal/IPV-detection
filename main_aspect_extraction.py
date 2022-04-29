# -*- coding: utf-8 -*-
"""
Created on Mon Apr  26 09:44:18 2022

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
from trainer.dl_trainer import Trainer
from utilities import utils
from utilities.read_configuration import DotDict

# Determinism.
SEED = 1234
random.seed(SEED)
rdm.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
