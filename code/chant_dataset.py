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

import os
import re
import math
import numpy as np
import torch.utils.data as data

from collections import Counter

import json

import pandas as pd
from pycantus import Cantus


class ChantDataset(data.Dataset):

    def __init__(self, seq_length, representation, target, traintest, split=0.8):
        self._seq_length = seq_length
        self._representation = representation
        self._target = target
        self._traintest = traintest

        if not os.path.isfile('data/stripped_'+ representation +'_data.json'):
            print('No converted data found, converting now')
            path = os.path.join(os.getcwd(), 'data/stripped_data.json')
            with open(path, 'r') as fp:
                d = json.load(fp)
            self._ids = d['ids']
            self._modes = d['modes']
            self._vps = d['vps']

            l = []
            tot = len(self._vps)
            for i,vp in enumerate(self._vps):
                print(i,'/',tot)
                l.append(self.extract_chars(vp))

            self._vps = l

            d['vps'] = self._vps
            
            path = os.path.join(os.getcwd(), 'data/stripped_'+ representation +'_data.json')
            with open(path, 'w') as fp:
                json.dump(d, fp)

        else:
            path = os.path.join(os.getcwd(), 'data/stripped_'+ representation +'_data.json')
            with open(path, 'r') as fp:
                d = json.load(fp)
            self._ids = d['ids']
            self._modes = d['modes']
            self._vps = d['vps']

        inds = [i for i, x in enumerate(self._vps) if ((len(x) > (seq_length - 1)) and ("T" not in self._modes[i]))]
        self._ids = [self._ids[i] for i in inds]
        self._vps = [self._vps[i][:seq_length] for i in inds]
        self._modes = [self._modes[i] for i in inds]

        self._unique_modes = list(set(self._modes))
        self._mode_num = len(self._unique_modes)

        flat_vps = [item for sublist in self._vps for item in sublist if item]
        self._vocab = list(set(flat_vps))
        self._vocab.sort()

        self._char_to_ix = {ch: i for i, ch in enumerate(self._vocab)}
        self._ix_to_char = {i: ch for i, ch in enumerate(self._vocab)}

        self._mode_to_ix = {m: i for i, m in enumerate(self._unique_modes)}
        self._ix_to_mode = {i: m for i, m in enumerate(self._unique_modes)}

        self._data_size, self._vocab_size = len(self._modes), len(self._vocab)

        self._split_ind = math.floor(self._data_size * split)

        self._vps_train = self._vps[:self._split_ind]
        self._modes_train = self._modes[:self._split_ind]
        self._ids_train = self._ids[:self._split_ind]

        self._vps_test = self._vps[self._split_ind:]
        self._modes_test = self._modes[self._split_ind:]
        self._ids_test = self._ids[self._split_ind:]

        print("Initialize dataset with {} training chants, {} test chants, with vocab size of {} and {} modes.".format(
            len(self._modes_train), len(self._modes_test), self._vocab_size, len(self._unique_modes)))

    def __getitem__(self, item):
        if self._traintest == 'train':
            inputs = [self._char_to_ix[ch] for ch in self._vps_train[item]]
            if self._target == 'mode':
                targets = self._mode_to_ix[self._modes_train[item]]

        if self._traintest == 'test':
            inputs = [self._char_to_ix[ch] for ch in self._vps_test[item]]
            if self._target == 'mode':
                targets = self._mode_to_ix[self._modes_test[item]]

        return inputs, targets

    def __len__(self):
        if self._traintest == 'train':
            return len(self._modes_train)
        else:
            return len(self._modes_test)

    def get_id(self, item):
        return self._ids[item]

    def get_test_set(self):
        inputs = []
        targets = []
        for i, vp in enumerate(self._vps_test):
            inputs.append([self._char_to_ix[ch] for ch in vp])
            targets.append(self._mode_to_ix[self._modes[i]])
        return inputs, targets

    def extract_chars(self, vp):
        if self._representation == 'raw':
            return [ch for ch in vp]
        elif self._representation == 'neume':
            return [i.strip("-") for i in vp.split("-") if i and i.strip("-") != '1']
        elif self._representation == 'syl':
            return [i.strip("-") for i in vp.split("--") if i and i.strip("-") != '1']
        elif self._representation == 'word':
            return [i.strip("-") for i in vp.split("---") if i and i.strip("-") != '1']