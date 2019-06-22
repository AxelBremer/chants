from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import argparse

import numpy as np
import json
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from itertools import combinations


from model import ModeModel
from chant_dataset import ChantDataset

def main(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = ChantDataset(seq_length=config.seq_length, representation=config.representation, target=config.target, traintest='train', notes=config.notes)

    vocab_size = dataset._vocab_size

    path = 'output/' + config.name    
    model_save_string = path + '/model.pt'

    if os.path.isfile(model_save_string):
        print('Loading model')
        model = torch.load(model_save_string, map_location=config.device)

    idx = list(range(vocab_size))
    combs = combinations(idx, config.num)

    for comb in combs:
        print(dataset.convert_to_string(comb))

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--seq_length', type=int, default=20, help='Length of an input sequence')
    parser.add_argument('--num', type=int, default=2, help='Length of combos to check')
    parser.add_argument('--device', type=str, default="cuda", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--name', type=str, default="debug", help="Name of the run")
    parser.add_argument('--representation', type=str, required=True, help="representation of the volpiano")
    parser.add_argument('--notes', type=str, default='interval', help="pitch or interval")
    parser.add_argument('--target', type=str, default='next', help="target [next] note or [mode]")

    config = parser.parse_args()

    # Train the model
    main(config)