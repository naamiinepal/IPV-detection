# -*- coding: utf-8 -*-
"""
Created on Wed Apr 6 20:03:45 2022

@author: Sagun Shakya

Description:
    Models used in training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class RNN(nn.Module):
    def __init__(self, config, dataloader):
        super().__init__()
        self.config = config
        self.bidirectional = config.rnn.bidirection
        self.num_layers = config.rnn.num_layers
        self.hidden_dim = config.rnn.hidden_dim
        self.embedding_dim = config.embedding_dim
        self.train_type = config.train_type if config.train_type is not None else 'text'
        
        self.vocab_size = dataloader.vocab_size
        self.tagset_size = dataloader.tagset_size

        # To use: a) Pre-trained Word Embedding b) Parameterized Word Embedding.
        if config.pretrained:
            self.embedding = nn.Embedding.from_pretrained(dataloader.weights, freeze = self.config.freeze_embedding)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)   
            
            
        #self.ac_size = dataloader.ac_size
        #self.ac_embeddings = nn.Embedding(self.ac_size, config.embedding_dim)
                  
        if config.model == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim, 
                                self.hidden_dim, 
                                num_layers = self.num_layers, 
                                bidirectional = self.bidirectional)
        elif config.model == 'gru':
            self.rnn = nn.GRU(self.embedding_dim, 
                                self.hidden_dim, 
                                num_layers = self.num_layers, 
                                bidirectional = self.bidirectional)            
        
        self.fc = nn.Linear(self.hidden_dim * 2, self.tagset_size)
        
        self.dropout = nn.Dropout(config.rnn.dropout)
        
    def forward(self, text):                # TAKE IN AC AND AT LATER.
        text = text.permute(1, 0)   # BEFORE: (batch_size, sent_len) --> AFTER: (sent_len, batch_size)
        #at = at.permute(1, 0)
        #ac = ac.permute(1, 0)        
        
        embedded = self.embedding(text)
        #at_emb = self.embedding(at)
        #ac_emb = self.ac_embeddings(ac)
        
        #embedded = torch.cat([embedded, ac_emb], dim=0)    
        
        # Only concatenate text and aspect term.
        #if self.train_type == 3:
        #    embedded = torch.cat((embedded, at_emb), dim=0)    # Shape -> (sent len (embedded+aspect_emb), batch size, emb dim)
        
        embedded, (hidden, cell) = self.rnn(self.dropout(embedded))
                
        # embedded = [batch size, sent_len, num_dim * hidden_dim]
        # hidden = [num_dim, sent_len, hidden_dim]

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)    # Shape -> (batch size, hid dim * num directions)
    
        final = self.fc(hidden)
        
        return final



class CNN(nn.Module):
    def __init__(self, config, dataloader):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.train_type = config.train_type if config.train_type is not None else 'text'
        self.num_filters = config.cnn.num_filters
        
        # Because filter_sizes are passed as string
        self.filter_sizes = tuple(map(int, config.cnn.filter_sizes.split(',')))
        
        self.vocab_size = dataloader.vocab_size
        self.tagset_size = dataloader.tagset_size

        # To use: a) Pre-trained Word Embedding b) Parameterized Word Embedding.
        if config.pretrained:
            self.embedding = nn.Embedding.from_pretrained(dataloader.weights, freeze = self.config.freeze_embedding)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)  
        
        #self.ac_size = dataloader.ac_size
        #self.ac_embeddings = nn.Embedding(self.ac_size, self.embedding_dim)
    
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = self.num_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in self.filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, self.tagset_size)
        
        self.dropout = nn.Dropout(config.cnn.dropout)
        
    def forward(self, text):
        
        #text = [batch size, seq_len] 
        # If the length of the sequence is less than the max of the kernel size (i.e max of [2,3,4] ==> 4), we pad the input tensor by 0.
        seq_len = text.shape[-1]
        limit = max(self.filter_sizes)
        if seq_len < limit:
            pad_sizes = (2,2,0,0) if seq_len == 1 else (1,1,0,0)
            pad = nn.ZeroPad2d(pad_sizes)      # Size of (left, right, top, bottom) padding.
            text = pad(text)
        
        assert text.shape[-1] >= limit, f"The length of the given sequence is {text.shape[-1]} which is less than the kernel size {limit}."
        
        embedded = self.embedding(text)     # Shape : (batch_size, seq_len, emb_dim)
        #at_emb = self.embedding(at)         # Shape : (batch_size, seq_len, emb_dim)
        #ac_emb = self.ac_embeddings(ac)     # Shape : (batch_size, 1, emb_dim)
        
        
        #if self.train_type == 2:
        #    embedded = torch.cat([embedded, ac_emb], dim=1)     # Shape : (batch_size, seq_len + 1, emb_dim)
        
        # Concatenate text and aspect term
        #elif self.train_type == 3:
        #    embedded = torch.cat([embedded, ac_emb], dim=1)
        #    embedded = torch.cat((embedded, at_emb), dim=1)
        
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)                        # Shape : (batch_size, 1, seq_len, emb_dim)
        
        conved = [F.relu(conv(self.dropout(embedded))).squeeze(3) for conv in self.convs]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = torch.cat(pooled, dim = 1)                        # Shape : (batch_size, 300)

        #cat = [batch size, n_filters * len(filter_sizes)]
        
        #final = F.softmax(self.fc(cat), dim=-1)
        
        final = self.fc(cat)                                    # Shape : (batch_size, 2)
        return final

class BertClassifier_LSTM(nn.Module):
    '''
    - pooled_output : (batch_size, bert_output_dim = 768)
    - sequence_output : (batch_size, seq_len = 512, bert_output_dim = 768)
    - lstm_out : (batch_size, 2 * hidden_dim)  
    - fc_out : (batch_size, output_dim = 2)
    
    Note: Apply softmax outside the classifier object.
    - output_probabilities = fc_out.argmax(dim = 1) 
    '''

    def __init__(self, config):

        super(BertClassifier_LSTM, self).__init__()
        self.config = config
        self.BERT_MODEL_NAME = self.config.bert.model_name
        self.n_layers = self.config.rnn.num_layers
        self.bidirectional = self.config.rnn.bidirection
        self.hidden_dim = self.config.rnn.hidden_dim
        self.output_dim = 2
        self.dropout = self.config.rnn.dropout
        
        self.bert = BertModel.from_pretrained(self.BERT_MODEL_NAME)
        self.embedding_dim = self.bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.LSTM(self.embedding_dim,
                          self.hidden_dim,
                          num_layers = self.n_layers,
                          bidirectional = self.bidirectional,
                          batch_first = True,
                          dropout = 0 if self.n_layers < 2 else self.dropout)

        self.fc = nn.Linear(self.hidden_dim * 2 if self.bidirectional else self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input_id, mask):

        sequence_output, pooled_output = self.bert(input_ids = input_id, attention_mask = mask, return_dict=False)
        # sequence_output shape : (batch_size, num_tokens = 512, output_dim = 768)
                
        _, (hidden, cell) = self.rnn(sequence_output)
        # hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])   # Shape: [batch size, hid dim]
        
        # hidden shape : (batch_size, emb_dim = 256 * 2 = 512)
        
        output = self.fc(hidden)     
        # Shape : [batch size, out dim]
        
        return output

class BertClassifier_Linear(nn.Module):

    def __init__(self, BERT_MODEL_NAME, output_dim = 2, dropout=0.5):

        super(BertClassifier_Linear, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input_id, mask):
        with torch.no_grad():
            _, pooled_output = self.bert(input_ids = input_id, attention_mask = mask, return_dict = False)

        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output
        