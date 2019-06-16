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
from tqdm import tqdm
from pycantus import Cantus
from pycantus import to_intervals, volpiano_characters, volpiano_to_midi

_VOLPIANO_TO_MIDI = {
    "8": 53, # F
    "9": 55, # G
    "a": 57,
    "y": 58, # B flat
    "b": 59,
    "c": 60,
    "d": 62,
    "w": 63, # E flat
    "e": 64,
    "f": 65,
    "g": 67,
    "h": 69,
    "i": 70, # B flat
    "j": 71,
    "k": 72, # C
    "l": 74,
    "x": 75, # E flat
    "m": 76,
    "n": 77,
    "o": 79,
    "p": 81,
    "z": 82, # B flat
    "q": 83, # B
    "r": 84, # C
    "s": 86,
    
    # Liquescents
    "(": 53,
    ")": 55,
    "A": 57,
    "B": 59,
    "C": 60,
    "D": 62,
    "E": 64,
    "F": 65,
    "G": 67,
    "H": 69,
    "J": 71,
    "K": 72, # C
    "L": 74,
    "M": 76,
    "N": 77,
    "O": 79,
    "P": 81,
    "Q": 83,
    "R": 84, # C
    "S": 86, # D
    
    # Naturals
    "Y": 59, # Natural at B
    "W": 64, # Natural at E
    "I": 71, # Natural at B
    "X": 76, # Natural at E
    "Z": 83,
}

def vp_to_midi(volpiano, skip_accidentals=False):
    """
    Translates volpiano pitches to a list of midi pitches

    All non-note characters are ignored or filled with `None`, if `fill_na=True`
    Unless `skip_accidentals=True`, accidentals are converted to midi pitches
    as well. So an i (flat at the B) becomes 70, a B flat. Or a W (a natural at
    the E) becomes 64 (E).
    """
    accidentals = volpiano_characters('flats', 'naturals')
    midi = []
    for char in volpiano:
        if skip_accidentals and char in accidentals:
            pass
        elif char in _VOLPIANO_TO_MIDI:
            midi.append(_VOLPIANO_TO_MIDI[char])
        elif char == '-':
            midi.append(char)
    return midi

class ChantDataset(data.Dataset):

    def __init__(self, seq_length, representation, target, traintest, notes, split=0.8):
        self._seq_length = seq_length
        self._representation = representation
        self._target = target
        self._traintest = traintest
        self._notes = notes

        if not os.path.isfile('data/'+ notes +'_stripped_'+ representation +'_data.json'):
            print('No converted data found, converting now')
            path = os.path.join(os.getcwd(), 'data/stripped_data.json')
            with open(path, 'r') as fp:
                d = json.load(fp)
            self._ids = d['ids']
            self._modes = d['modes']
            self._vps = d['vps']

            l = []
            for vp in tqdm(self._vps):
                l.append(self.extract_chars(vp))

            self._vps = l

            d['vps'] = self._vps
            
            path = os.path.join(os.getcwd(), 'data/'+ notes +'_stripped_'+ representation +'_data.json')
            with open(path, 'w') as fp:
                json.dump(d, fp)

        else:
            path = os.path.join(os.getcwd(), 'data/'+ notes +'_stripped_'+ representation +'_data.json')
            with open(path, 'r') as fp:
                d = json.load(fp)
            self._ids = d['ids']
            self._modes = d['modes']
            self._vps = d['vps']

        # inds = [i for i, x in enumerate(self._vps) if ((len(x) > (seq_length - 1)))]
        # self._ids = [self._ids[i] for i in inds]
        # self._vps = [self._vps[i][:seq_length] for i in inds]
        # self._modes = [self._modes[i] for i in inds]
        inds = []
        i = []
        v = []
        m = []

        for j, x in tqdm(enumerate(self._vps)):
            if len(x) > (seq_length - 1):
                i.append(self._ids[j])
                v.append(x)
                m.append(self._modes[j])

        self._ids = i
        self._vps = v
        self._modes = m
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
            if self._notes == 'interval':
                return to_intervals(volpiano_to_midi(vp))
            elif self._notes == 'pitch':
                l = [ch for ch in vp]
            return l
        elif self._representation == 'neume':
            if self._notes == 'interval':
                return get_intervals(vp, 1)
            elif self._notes == 'pitch':
                l =  [i.strip("-") for i in vp.split("-") if i and i.strip("-") != '1']
                return l
        elif self._representation == 'syl':
            if self._notes == 'interval':
                return get_intervals(vp, 1)
            elif self._notes == 'pitch':
                l = [i.strip("-") for i in vp.split("--") if i and i.strip("-") != '1']
                return l
        elif self._representation == 'word':
            if self._notes == 'interval':
                return get_intervals(vp, 1)
            elif self._notes == 'pitch':
                l = [i.strip("-") for i in vp.split("---") if i and i.strip("-") != '1']
                return l

def get_intervals(vp, count):
    inter = to_intervals(volpiano_to_midi(vp))
    midi = vp_to_midi(vp)
    inter[0] = 0
    inter = list(reversed(inter))
    l = nSplit(midi, '-', 1)
    for i,n in enumerate(l):
        s = ''
        for ch in n:
            s = s+str(inter.pop())
        l[i] = s
    return l

def nSplit(lst, delim, count=2):
    output = [[]]
    delimCount = 0
    for item in lst:
        if item == delim:
            delimCount += 1
        elif delimCount >= count:
            output.append([item])
            delimCount = 0
        else:
            output[-1].append(item)
            delimCount = 0
    return output[1:]

# d = ChantDataset(20, 'word', 'mode', 'train', 'interval')
# print(d[0])