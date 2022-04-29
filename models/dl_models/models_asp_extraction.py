# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 07:13:45 2022

@author: Sagun Shakya

Description:
    Models used in Aspect Term Extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel