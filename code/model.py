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

    def __init__(self, batch_size, seq_length, mode_num,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(ModeModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.mode_num
        self.device = device
        self.steps = 0

        self.lstm = nn.LSTM(input_size = self.vocabulary_size,
                            hidden_size = self.lstm_num_hidden,
                            num_layers = self.lstm_num_layers,
                            batch_first = True)

        self.linear = nn.Linear(in_features = self.lstm_num_hidden,
                                out_features = self.mode_num,
                                bias = True)

    def forward(self, x, states=None):
        if states != None:
            lstm_out, states = self.lstm(x, states)
        else:
            lstm_out, states = self.lstm(x)
        out = self.linear(lstm_out)
        return out, states

    def step(self):
        self.steps += 1
