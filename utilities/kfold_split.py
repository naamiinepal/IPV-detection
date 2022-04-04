# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 07:44:32 2022

@author: Sagun Shakya
"""
import os
from os.path import join
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split, StratifiedKFold

# Local Modules.
from utilities import utils

# Parameters.
random_seed = 1234
root_raw = r'data/raw'
save_path = r'data/kfold'

#%% Read raw data.

# Load IPV examples.
ipv = pd.read_csv(join(root_raw, 'IPV_sents.tsv'), delimiter = '\t', encoding = 'utf-8')
ipv.columns = "id sents".split()
ipv['pol'] = np.ones(len(ipv), dtype = np.int8)

# Load IPV examples.
non_ipv = pd.read_csv(join(root_raw, 'non-IPV_sents.tsv'), delimiter = '\t', encoding = 'utf-8')
non_ipv.columns = "id sents".split()
non_ipv['pol'] = np.zeros(len(non_ipv), dtype = np.int8)

#%% Verbose.
print("Number of IPV instances : ", len(ipv))
print("Number of non-IPV instances : ", len(non_ipv))

#%% Concat both.
df = pd.concat([ipv, non_ipv], axis = 0).sample(frac = 1, random_state = random_seed)
df.reset_index(drop = True, inplace = True)

#%% GroupShuffle Split.
# Split the df based on polarity labels.
gss = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = random_seed).split(df, groups=df['pol'])

# Get positive and negative dataframe
for positive_df, negative_df in gss:
    
    # Get data based on the index.
    negative = df.iloc[negative_df]
    positive = df.iloc[positive_df]
    
    print(negative)
    print(positive)
    
    # Split 80/10/10 -> train, test, val based on polarity.
    train_neg, test_val_neg = train_test_split(negative, test_size=0.2, random_state = random_seed)
    train_pos, test_val_pos = train_test_split(positive, test_size=0.2, random_state = random_seed)
    test_neg, val_neg = train_test_split(test_val_neg, test_size=0.5, random_state = random_seed)
    test_pos, val_pos = train_test_split(test_val_pos, test_size=0.5, random_state = random_seed)
    
    # Concat negative and positive dataframe and shuffle
    train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)
    val_df = pd.concat([val_pos, val_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)
    
    #utils.write_csv(df, target_filename)
    
#%% Stratified kfold.
class StratifiedKFold3(StratifiedKFold):
    '''
    Performs stratified 5-fold.
    Returns: 
        Train indices, val indices, test indices.
    '''
    def split(self, X, y, groups=None):
        s = super().split(X, y, groups)
        for train_indxs, test_indxs in s:
            y_train = y[train_indxs]
            train_indxs, cv_indxs = train_test_split(train_indxs,stratify=y_train, test_size=0.20)
            yield train_indxs, cv_indxs, test_indxs

# Instantiate.
gg = StratifiedKFold3(5).split(df['sents'].values, df['pol'].values)

# Write Results.
for k, (train_id, val_id, test_id) in enumerate(gg, 1):
    train_df = df.iloc[train_id, :]
    val_df = df.iloc[val_id, :]
    test_df = df.iloc[test_id, :]
    
    save_filepath = join(save_path, str(k))
    os.makedirs(save_filepath, exist_ok=True)
    
    utils.write_csv(train_df, join(save_filepath, 'train.txt'))
    utils.write_csv(val_df, join(save_filepath, 'val.txt'))
    utils.write_csv(test_df, join(save_filepath, 'test.txt'))

#%% Verbose.
train_coll = {'0' : [], '1' : []}
val_coll = {'0' : [], '1' : []}
test_coll = {'0' : [], '1' : []}


for k in range(1, 6):
    save_filepath = join(save_path, str(k))
    train_df = pd.read_csv(join(save_filepath, 'train.txt'), header = None, encoding = 'utf-8')
    val_df = pd.read_csv(join(save_filepath, 'val.txt'), header = None, encoding = 'utf-8')
    test_df = pd.read_csv(join(save_filepath, 'test.txt'), header = None, encoding = 'utf-8')
    
    train_dict = train_df.iloc[:, -1].value_counts().to_dict()
    val_dict = val_df.iloc[:, -1].value_counts().to_dict()
    test_dict = test_df.iloc[:, -1].value_counts().to_dict()
    
    train_coll['0'].append(train_dict[0])
    train_coll['1'].append(train_dict[1])

    val_coll['0'].append(val_dict[0])
    val_coll['1'].append(val_dict[1])

    test_coll['0'].append(test_dict[0])
    test_coll['1'].append(test_dict[1])
    
print("Size of train data across 5 folds : \n", train_coll)
print("Size of validation data across 5 folds : \n", val_coll)
print("Size of test data across 5 folds : \n", test_coll)