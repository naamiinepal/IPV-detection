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

class RNN(nn.Module):
    def __init__(self, config, dataloader):
        super().__init__()
        self.config = config
        self.bidirectional = config.lstm.bidirection
        self.num_layers = config.lstm.num_layers
        self.hidden_dim = config.lstm.hidden_dim
        self.vocab_size = dataloader.vocab_size
        self.tagset_size = dataloader.tagset_size
        self.embedding_dim = config.embedding_dim
        self.train_type = config.train_type if config.train_type is not None else 'text'
        
        # To use: a) Pre-trained Word Embedding b) Parameterized Word Embedding.
        if config.pretrained:
            self.embedding = nn.Embedding.from_pretrained(dataloader.weights)
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
        
        self.dropout = nn.Dropout(config.dropout)
        
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
        self.hidden_dim = config.hidden_dim
        self.vocab_size = dataloader.vocab_size
        self.tagset_size = dataloader.tagset_size
        self.embedding_dim = config.embedding_dim
        self.train_type = config.train_type
        self.num_filters = config.num_filters
        
        # Because filter_sizes are passed as string
        self.filter_sizes = [int(x) for x in config.filter_sizes.split(',')]
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.ac_size = dataloader.ac_size
        self.ac_embeddings = nn.Embedding(self.ac_size, self.embedding_dim)
    
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = self.num_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in self.filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, self.tagset_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, text, at, ac):
        
        #text = [sent len, batch size]      
        
        embedded = self.embedding(text)     # Shape : (batch_size, seq_len, emb_dim)
        at_emb = self.embedding(at)         # Shape : (batch_size, seq_len, emb_dim)
        ac_emb = self.ac_embeddings(ac)     # Shape : (batch_size, 1, emb_dim)
        
        
        if self.train_type == 2:
            embedded = torch.cat([embedded, ac_emb], dim=1)     # Shape : (batch_size, seq_len + 1, emb_dim)
        
        # Concatenate text and aspect term
        elif self.train_type == 3:
            embedded = torch.cat([embedded, ac_emb], dim=1)
            embedded = torch.cat((embedded, at_emb), dim=1)
        
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)                        # Shape : (batch_size, 1, seq_len + 1, emb_dim)
        
        conved = [F.relu(conv(self.dropout(embedded))).squeeze(3) for conv in self.convs]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = torch.cat(pooled, dim = 1)                        # Shape : (batch_size, 300)

        #cat = [batch size, n_filters * len(filter_sizes)]
        
        #final = F.softmax(self.fc(cat), dim=-1)
        
        final = self.fc(cat)                                    # Shape : (batch_size, 2)
        return final