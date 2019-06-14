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
import numpy as np
import torch.utils.data as data

from collections import Counter

import json

import pandas as pd
from pycantus import Cantus


class ChantDataset(data.Dataset):

    def __init__(self, seq_length=15, representation='raw'):
        path = os.path.join(os.getcwd(), 'data/stripped_data.json')
        with open(path, 'r') as fp:
            d = json.load(fp)
        self._ids = d['ids']
        self._modes = d['modes']
        self._vps = d['vps']

        self.representation = representation

        self._seq_length = seq_length

        inds = [i for i, x in enumerate(self._vps) if ((len(x) > (seq_length - 1)) and ("T" not in self._modes[i]))]
        self._ids = [self._ids[i] for i in inds]
        self._vps = [self._vps[i][:seq_length] for i in inds]
        self._modes = [self._modes[i] for i in inds]

        self._unique_modes = list(set(self._modes))
        self._mode_num = len(self._unique_modes)

        if representation == 'raw':
            self._vocab = list(set(''.join(self._vps)))
        elif representation == 'neume':
            # neumes vocabulary
            self._neum_list = [[i.strip("-") for i in string.split("-") if i and i.strip("-") != '1'] for string in self._vps]
            self._flat_neum = [item for self._sublist in self._neum_list for item in self._sublist if item]
            self._vocab =  list(set(self._flat_neum))
        elif representation == 'syl':
            # syllables vocabulary
            self._syll_list = [[i.strip("-") for i in string.split("--") if i and i.strip("-") != '1'] for string in
                            self._vps]
            self._flat_syll = [item for self._sublist in self._syll_list for item in self._sublist if item]
            self._vocab =  list(set(self._flat_syll))
        elif representation == 'word':
            # words vocabulary
            self._word_list = [[i.strip("-") for i in string.split("---") if i and i.strip("-") != '1'] for string in self._vps]
            self._flat_words = [item for self._sublist in self._word_list for item in self._sublist if item]
            self._vocab =  list(set(self._flat_words))


        self._vocab.sort()

        self._char_to_ix = {ch: i for i, ch in enumerate(self._vocab)}
        self._ix_to_char = {i: ch for i, ch in enumerate(self._vocab)}

        self._mode_to_ix = {m: i for i, m in enumerate(self._unique_modes)}
        self._ix_to_mode = {i: m for i, m in enumerate(self._unique_modes)}

        self._data_size, self._vocab_size = len(self._modes), len(self._vocab)
        print("Initialize dataset with {} chants, with vocab size of {} and {} modes.".format(
            self._data_size, self._vocab_size, len(self._unique_modes)))


        # print("Neumes vocab \n {}. \n\n Syllables vocab \n {}, \n \n Words vocab \n {}".format(
            # self._neumes, self._syllables, self._words))

    def __getitem__(self, item):
        inputs = [self._char_to_ix[ch] for ch in self.extract_chars(self._vps[item])]
        targets = self._mode_to_ix[self._modes[item]]
        return inputs, targets

    def __len__(self):
        return self._data_size

    def get_id(self, item):
        return self._ids[item]

    def extract_chars(self, vp):
        if self.representation == 'raw':
            return [ch for ch in vp]
        elif self.representation == 'neume':
            return [i.strip("-") for i in vp.split("-") if i and i.strip("-") != '1']
        elif self.representation == 'syl':
            return [i.strip("-") for i in vp.split("--") if i and i.strip("-") != '1']
        elif self.representation == 'word':
            return [i.strip("-") for i in vp.split("---") if i and i.strip("-") != '1']


# d = ChantDataset(60, 'neume')
# vp = '1--d--d--dfd-dc---f---g--ghgf-ghg-hj--h'
# print(vp)
# print(d.extract_chars(vp))