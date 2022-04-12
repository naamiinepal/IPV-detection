# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:44:35 2022

@author: Sagun Shakya
"""

# Necessary libraries.
import pandas as pd
import copy
import os
from os.path import join, exists

# Local modules.
from .annotation_utils import get_file, convert_to_bio

# Path to data.
root = r'annotation\data'
annot1 = r'annotation\data\krn'
annot2 = r'annotation\data\shr'

example = 'batch_ipv_600-610 exported.tsv'

def run():
    filename = join(annot2, example)

    # Getting the exported TSV file.
    text = get_file(filename)
    text = [tt for tt in text if tt != '\n']
    text = text[3:]
    #print(text[:6])

    # Save.
    storage = []
    count = 0    # Sentence counter.
    for ii, line in enumerate(text, 1):
        if line.startswith('#Text'):
            count += 1
            continue
        else:
            storage.append(line.split())
    
    print(storage[:10])
if __name__ == "__main__":
    run()
