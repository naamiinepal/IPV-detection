# Libraries.
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from datetime import datetime


def classification_metrics(Y_true, Y_pred):
    '''
    This function displays the accuracy score, f1 - score and the ROC_AUC score
    for given classification predictions and the true labels.
    '''
    acc = accuracy_score(Y_true, Y_pred)
    pr = precision_score(Y_true, Y_pred)
    rec = recall_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred)
    auc = roc_auc_score(Y_true, Y_pred)

    return acc, pr, rec, f1, auc

def verbosity(acc, pr, rec, f1, auc, mode = 'train'):
    '''
    Displays the Accuracy Score, Precision, Recall and F1 Score and ROC-AUC Score for train, val or test set.
    '''
    assert mode == 'train' or mode == 'val' or mode == 'test', "mode should be one of ('train', 'val', 'test')."
    
    verbose = f'Results for {mode} set'

    print('_'*len(verbose))
    print(verbose)
    print('_'*len(verbose))
    
    print(f'Accuracy : {acc:0.3f}')
    print(f'Precision : {pr:.3f}  ||  Recall : {rec:.3f}  ||  F1 Score : {f1:.3f}')
    print(f'ROC-AUC Score : {auc:.3f}\n')

def train_time(start_time, end_time):
    '''
    Time taken for the training and evaluation to complete.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def current_timestamp():
    '''
    Current date and time.

    Returns
    -------
    str
        date and time.

    '''
    dateTimeObj = datetime.now()
    date = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + '-' + str(dateTimeObj.day)
    time = str(dateTimeObj.hour) + ':' + str(dateTimeObj.minute) + ':' + str(dateTimeObj.second)
    
    return f'{date} || {time}'