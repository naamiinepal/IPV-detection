# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:33:19 2022

@author: Sagun Shakya
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

tqdm.pandas(desc="Progress")


class Evaluator:
    def __init__(self, config, logger, model, dataloader, model_name):
        self.config = config
        self.logger = logger
        self.model = model
        self.model_name = model_name
        self.dataloader = dataloader

        self.train_dl, self.val_dl, self.test_dl = dataloader.load_data(
            batch_size=config.batch_size, shuffle=False
        )
        self.results_dir = config.results_dir

        ts_file = self.model_name + "_test.txt"
        self.test_file = os.path.join(self.results_dir, ts_file)

        self.average = config.average

    def numpy_to_sent(self, tensor):
        """
        Returns the corresponding TEXT of given Predictions
        Returns chunks of string
        """
        return " ".join(
            [
                self.dataloader.txt_field.vocab.itos[i]
                for i in tensor.cpu().data.numpy()[0]
            ]
        ).split()

    def numpy_to_at(self, tensor):
        """
        Returns the corresponding ASPECT TERM of given Predictions
        Returns chunks of string
        """
        return " ".join(
            [
                self.dataloader.at_field.vocab.itos[i]
                for i in tensor.cpu().data.numpy()[0]
            ]
        ).split()

    def numpy_to_ac(self, tensor):
        """
        Returns the corresponding ASPECT TERM of given Predictions
        Returns chunks of string
        """
        return " ".join([self.dataloader.ac_field.vocab.itos[i] for i in tensor])

    def pred_to_tag(self, predictions):
        """
        Returns the corresponding LABEL of given Predictions
        Returns chunks of string
        """
        if self.config.train_type == 3 or self.config.train_type == 4:
            return " ".join(
                [self.dataloader.ss_field.vocab.itos[i] for i in predictions]
            )
        else:
            return " ".join(
                [self.dataloader.ac_field.vocab.itos[i] for i in predictions]
            )

    def write_results(self):
        """
        Writes the result into the file
        """
        self.model.eval()

        with torch.no_grad() and open(self.test_file, "w", encoding="utf-8") as rtst:
            self.logger.info(f"Writing the results in file: {self.test_file}")
            tt = tqdm(iter(self.test_dl), leave=False)

            for ((y, ac, at, X), v) in tt:

                pred = self.model(X, at, ac)
                pred = F.softmax(pred, dim=-1)

                for i in range(X.shape[0]):
                    txt = X[i].unsqueeze(0)
                    aterm = at[i].unsqueeze(0)
                    acat = ac[i].unsqueeze(0)
                    gold = y[i].unsqueeze(0)
                    predicted = pred[i].unsqueeze(0)

                    sent = self.numpy_to_sent(txt)
                    sent = " ".join(sent)

                    aspect_cat = self.numpy_to_ac(acat)

                    aspect = self.numpy_to_at(aterm)
                    aspect = " ".join(aspect)

                    y_true_val = gold.squeeze(1).data.cpu().numpy()
                    true_tag = self.pred_to_tag(y_true_val)

                    y_pred_val = (
                        predicted.argmax(dim=1, keepdim=True)
                        .squeeze(1)
                        .data.cpu()
                        .numpy()
                    )
                    pred_tag = self.pred_to_tag(y_pred_val)

                    rtst.write(
                        sent
                        + "\t"
                        + aspect
                        + "\t"
                        + aspect_cat
                        + "\t"
                        + true_tag
                        + "\t"
                        + pred_tag
                        + "\n"
                    )

                    rtst.write("\n")

            rtst.close()

    def infer(self, sent, aspect_term, aspect_cat):
        """
        Prints the result
        """
        # Tokenize the sentence and aspect terms.
        sent_tok = self.dataloader.tokenizer(sent)
        at_tok = self.dataloader.tokenizer(aspect_term)

        # Words to indices.
        X = [self.dataloader.txt_field.vocab.stoi[t] for t in sent_tok]
        at = [self.dataloader.at_field.vocab.stoi[t] for t in at_tok]
        ac = [self.dataloader.ac_field.vocab.stoi[aspect_cat]]

        # Convert into torch long tensor and reshape into [batch, sent_length]
        X = torch.LongTensor(X).to(self.config.device)
        X = X.unsqueeze(0)

        at = torch.LongTensor(at).to(self.config.device)
        at = at.unsqueeze(0)

        ac = torch.LongTensor(ac).to(self.config.device)
        ac = ac.unsqueeze(0)

        # Get predictions.
        pred = self.model(X, at, ac)
        pred = F.softmax(pred, dim=-1)

        pred_idx = pred.argmax(dim=1)

        y_pred_val = pred_idx.cpu().data.numpy()
        pred_tag = self.pred_to_tag(y_pred_val)
        return pred_tag

    def prec_rec_f1(self, gold_list, pred_list):
        """
        Calculates the Precision, Recall, F1-score and ROC-AUC Score.

        Parameters
        ----------
        gold_list : list
            true labels.
        pred_list : list
            predicted labels.

        Returns
        -------
        float.
        Precision, Recall, F1-score and ROC-AUC Score.

        """
        prec, rec, f1, _ = precision_recall_fscore_support(
            gold_list, pred_list, average=self.average
        )
        gold_list = np.array(gold_list)
        pred_list = np.array(pred_list)

        n_values = np.max(gold_list) + 1

        # create one hot encoding for auc calculation
        gold_list = np.eye(n_values)[gold_list]
        pred_list = np.eye(n_values)[pred_list]
        auc = roc_auc_score(gold_list, pred_list, average=self.average)
        return prec, rec, f1, auc
