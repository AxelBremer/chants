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
import torch
import torch.utils.data as data

from collections import Counter

import json
import pickle
import pandas as pd
from tqdm import tqdm
from pycantus import Cantus
from pycantus import to_intervals, volpiano_characters, volpiano_to_midi, get_interval_representation

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
        elif char in ['-']:
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
            self._genres = d['genres']

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
            self._genres = d['genres']


        inds = []
        i = []
        v = []
        m = []
        g = []

        for j, x in enumerate(self._vps):
            if x != None:
                if len(x) > (seq_length):
                    i.append(self._ids[j])
                    v.append(x)
                    m.append(self._modes[j])
                    g.append(str(self._genres[j]))

        self._ids = i
        self._vps = v
        self._modes = m
        self._genres = g
        self._unique_modes = list(set(self._modes))
        self._mode_num = len(self._unique_modes)
        self._unique_genres = list(set(self._genres))
        self._genre_num = len(self._unique_genres)

        flat_vps = [item for sublist in self._vps for item in sublist if item]
        self._vocab = list(set(flat_vps))
        self._vocab.sort()

        self._unique_genres.sort()
        self._unique_modes.sort()

        self._char_to_ix = {ch: i for i, ch in enumerate(self._vocab)}
        self._ix_to_char = {i: ch for i, ch in enumerate(self._vocab)}

        self._mode_to_ix = {m: i for i, m in enumerate(self._unique_modes)}
        self._ix_to_mode = {i: m for i, m in enumerate(self._unique_modes)}

        self._genre_to_ix = {m: i for i, m in enumerate(self._unique_genres)}
        self._ix_to_genre = {i: m for i, m in enumerate(self._unique_genres)}

        print(self._ix_to_genre)

        self._data_size, self._vocab_size = len(self._modes), len(self._vocab)

        self._split_ind = math.floor(self._data_size * split)

        v = []
        print('converting to indices')
        print(self._vocab_size)
        for vp in self._vps:
            v.append([self._char_to_ix[ch] for ch in vp[:self._seq_length+1]])
        self._vps = v

        self._vps_train = self._vps[:self._split_ind]
        self._modes_train = self._modes[:self._split_ind]
        self._ids_train = self._ids[:self._split_ind]
        self._genres_train = self._genres[:self._split_ind]

        self._vps_test = self._vps[self._split_ind:]
        self._modes_test = self._modes[self._split_ind:]
        self._ids_test = self._ids[self._split_ind:]
        self._genres_test = self._genres[self._split_ind:]

        inputpath = 'data/inputs/'+ notes +'_' + str(seq_length) + '_'+ representation 
        if not os.path.isfile(inputpath +'_string_genre_corpus_train.txt'):
            print('saving input to file')
            string_corpus = ['%'.join([self._ix_to_char[i] for i in vp[:-1]]) for vp in self._vps_train]
            input_corpus = ['%'.join([str(i) for i in vp[:-1]]) for vp in self._vps_train]

            mode_target_corpus = ['%'.join([str(self._mode_to_ix[mode]) for i in range(seq_length)]) for mode in self._modes_train]

            genre_target_corpus = ['%'.join([str(self._genre_to_ix[genre]) for i in range(seq_length)]) for genre in self._genres_train]
            
            mode_corpus = '\n'.join(f'{w}\t{t}' for w,t in zip(input_corpus, mode_target_corpus))

            with open(inputpath + '_mode_corpus_train.txt', 'w') as f:
                f.write(mode_corpus)

            string_mode_corpus = '\n'.join(f'{w}\t{t}' for w,t in zip(string_corpus, mode_target_corpus))
            string_genre_corpus = '\n'.join(f'{w}\t{t}' for w,t in zip(string_corpus, genre_target_corpus))

            with open(inputpath + '_string_mode_corpus_train.txt', 'wb') as f:
                f.write(string_mode_corpus.encode('utf-8'))
            with open(inputpath + '_string_genre_corpus_train.txt', 'wb') as f:
                f.write(string_genre_corpus.encode('utf-8'))

            with open(inputpath + '_genre_vocab.txt', 'w') as f:
                f.write('\n'.join([str(x) for x in self._unique_genres]))

            with open(inputpath + '_vocab.txt', 'wb') as f:
                voc = '\n'.join([x for x in self._vocab])
                f.write(voc.encode('utf-8'))

        if not os.path.isfile(inputpath +'_string_genre_corpus_test.txt'):
            print('saving input to file')
            string_corpus = ['%'.join([self._ix_to_char[i] for i in vp[:-1]]) for vp in self._vps_test]
            input_corpus = ['%'.join([str(i) for i in vp[:-1]]) for vp in self._vps_test]
            mode_target_corpus = ['%'.join([str(self._mode_to_ix[mode]) for i in range(seq_length)]) for mode in self._modes_test]

            genre_target_corpus = ['%'.join([str(self._genre_to_ix[genre]) for i in range(seq_length)]) for genre in self._genres_test]
            
            mode_corpus = '\n'.join(f'{w}\t{t}' for w,t in zip(input_corpus, mode_target_corpus))
            string_genre_corpus = '\n'.join(f'{w}\t{t}' for w,t in zip(string_corpus, genre_target_corpus))

            with open(inputpath + '_mode_corpus_test.txt', 'w') as f:
                f.write(mode_corpus)

            string_mode_corpus = '\n'.join(f'{w}\t{t}' for w,t in zip(string_corpus, mode_target_corpus))
            string_genre_corpus = '\n'.join(f'{w}\t{t}' for w,t in zip(string_corpus, genre_target_corpus))

            with open(inputpath + '_string_mode_corpus_test.txt', 'wb') as f:
                f.write(string_mode_corpus.encode('utf-8'))
            with open(inputpath + '_string_genre_corpus_test.txt', 'wb') as f:
                f.write(string_genre_corpus.encode('utf-8'))



    def __getitem__(self, item):
        if self._traintest == 'train':
            # inputs = [self._char_to_ix[ch] for ch in self._vps_train[item][:self._seq_length]]
            inputs = self._vps_train[item][:self._seq_length]
            mode_targets = self._mode_to_ix[self._modes_train[item]]
            next_targets = self._vps_train[item][1:self._seq_length+1]
            genre = self._genre_to_ix[self._genres_train[item]]

        if self._traintest == 'test':
            # inputs = [self._char_to_ix[ch] for ch in self._vps_test[item][:self._seq_length]]
            inputs = self._vps_test[item][:self._seq_length]
            mode_targets = self._mode_to_ix[self._modes_test[item]]
            next_targets = self._vps_test[item][1:self._seq_length+1]
            genre = self._genre_to_ix[self._genres_test[item]]

        return inputs, next_targets, mode_targets, genre

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
                return self.get_intervals(vp)
            elif self._notes == 'pitch':
                l = [ch for ch in vp]
            return l
        elif self._representation == 'neume':
            if self._notes == 'interval':
                vp = self.get_intervals(vp)
            l =  [i.strip("-") for i in vp.split("-") if i and i.strip("-") != '1']
            # if l[-1] == '': l = l[:-1]
            return l
        elif self._representation == 'syllable':
            if self._notes == 'interval':
                vp = self.get_intervals(vp)
            l = [i.strip("-") for i in vp.split("--") if i and i.strip("-") != '1']
            # if l[-1] == '': l = l[:-1]
            return l
        elif self._representation == 'word':
            if self._notes == 'interval':
                vp = self.get_intervals(vp)
            l = [i.strip("-") for i in vp.split("---") if i and i.strip("-") != '1']
            # if l[-1] == '': l = l[:-1]
            return l

    def get_intervals(self, vp):
        inter = to_intervals(volpiano_to_midi(vp), encode=True)
        midi = vp_to_midi(vp)
        if vp[-1] in ['3','4']: midi.append(vp[-1])
        s = ''
        ct = 0
        for i,n in enumerate(midi):
            if n not in ['-','3','4']:
                s += inter[ct]
                ct += 1
            else:
                s += n
        return s
    
    def convert_to_string(self, char_ix):
        if self._representation == 'raw':
            return ''.join(self._ix_to_char[ix] for ix in char_ix)
        if self._representation == 'neume':
            return '-'.join(self._ix_to_char[ix] for ix in char_ix)
        if self._representation == 'syllable':
            return '--'.join(self._ix_to_char[ix] for ix in char_ix)
        if self._representation == 'word':
            return '---'.join(self._ix_to_char[ix] for ix in char_ix)

def num2hot(batch, vocab_size):
    # Get the shape of the input and add the vocabulary size in a new dimension
    shape = list(batch.shape)
    shape = shape + [vocab_size]

    # Create the output tensor and use it as index to place a one in the new tensor
    y_out = torch.zeros(shape)
    batch = batch.unsqueeze(-1)

    y_out.scatter_(2, batch.long(), torch.tensor(1))

    return y_out

# d = ChantDataset(20, 'neume', 'next', 'train', 'pitch')
# d = ChantDataset(20, 'neume', 'next', 'train', 'interval')
# d = ChantDataset(20, 'raw', 'next', 'train', 'pitch')
# d = ChantDataset(20, 'raw', 'next', 'train', 'interval')
# d = ChantDataset(30, 'raw', 'next', 'train', 'pitch')
# d = ChantDataset(30, 'raw', 'next', 'train', 'interval')
# d = ChantDataset(20, 'syllable', 'next', 'train', 'pitch')
# d = ChantDataset(20, 'syllable', 'next', 'train', 'interval')