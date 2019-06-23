# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class ModeModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocab_size, mode_num, target,
                 lstm_num_hidden=256, lstm_num_layers=1, device='cuda:0'):

        super(ModeModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.vocab_size = vocab_size
        self.mode_num = mode_num
        self.device = device
        self.epochs = 0
        self.target = target

        self.lstm = nn.LSTM(input_size = self.vocab_size,
                                        hidden_size = self.lstm_num_hidden,
                                        num_layers = self.lstm_num_layers,
                                        batch_first = True)

        self.dropout = nn.Dropout(0.5)

        if target == 'mode':
            # self.linear = nn.Linear(in_features = self.lstm_num_hidden,
            #                         out_features = self.mode_num,
            #                         bias = True)
            self.linear = nn.Sequential(nn.Linear(in_features = self.lstm_num_hidden,
                                                  out_features = 512,
                                                  bias = True),
                                        nn.LeakyReLU(0.2),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features = 512,
                                                  out_features = self.mode_num,
                                                  bias = True))
        if target == 'next':
            self.linear = nn.Linear(in_features = self.lstm_num_hidden,
                                    out_features = self.vocab_size,
                                    bias = True)

        if target == 'both':
            self.mode_layer = nn.Linear(in_features = self.lstm_num_hidden,
                                        out_features = self.mode_num,
                                        bias = True)
            self.next_layer = nn.Linear(in_features = self.lstm_num_hidden,
                                    out_features = self.vocab_size,
                                    bias = True)

    def forward(self, x, states=None):
        if self.target == 'both':
            if states != None:
                lstm_out, states = self.lstm(x, states)
            else:
                lstm_out, states = self.lstm(x)

            mode_out = self.mode_layer(self.dropout(lstm_out))[:,-1,:].squeeze()
            next_out = self.next_layer(self.dropout(lstm_out))
            # return out[:,-1,:].squeeze(), states
            return next_out, mode_out, states
        else:
            if states != None:
                lstm_out, states = self.lstm(x, states)
            else:
                lstm_out, states = self.lstm(x)
            drop_out = self.dropout(lstm_out)
            out = self.linear(drop_out)
            # return out[:,-1,:].squeeze(), states
            if self.target == 'mode':
                return '', out[:,-1,:].squeeze(), states
            if self.target == 'next':
                return out, '', states

    def next_epoch(self):
        self.epochs += 1
