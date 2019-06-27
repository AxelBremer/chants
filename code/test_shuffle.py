from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import argparse

import numpy as np
import json
import math
import joblib
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from model import ModeModel
from chant_dataset import ChantDataset

def num2hot(batch, vocab_size, device):
    # Get the shape of the input and add the vocabulary size in a new dimension
    shape = list(batch.shape)
    shape = shape + [vocab_size]

    # Create the output tensor and use it as index to place a one in the new tensor
    y_out = torch.zeros(shape).to(device)
    batch = batch.unsqueeze(-1)

    y_out.scatter_(2, batch, torch.tensor(1).to(device))

    return y_out

def get_accuracy(y_target, y_pred, config):
    return (y_pred.argmax(dim=1) == y_target).sum().cpu().numpy().item()/(config.batch_size)


def train(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    path = 'output/' + config.name    
    model_save_string = path + '/model.pt'
    mode_model_save_string = path + '/classifiers/hx_l1.joblib'

    if os.path.isfile(model_save_string):
        print('Loading model')
        gen_model = torch.load(model_save_string, map_location=config.device)
        mode_model = joblib.load(mode_model_save_string)
    else:
        print('No gen model')
        exit(1)

    if 'raw' in config.name:
        representation = 'raw'
    elif 'neume' in config.name:
        representation = 'neume'
    elif 'syllable' in config.name:
        representation = 'syllable'

    if 'pitch' in config.name:
        notes = 'pitch'
    elif 'interval' in config.name:
        notes = 'interval'

    # # Initialize the dataset and data loader (note the +1)
    # dataset = ChantDataset(seq_length=gen_model.seq_length, representation=representation, target='mode', traintest='train', notes=notes)
    # data_loader = DataLoader(dataset, 128, num_workers=4)

    # Initialize the dataset and data loader (note the +1)
    test_dataset = ChantDataset(seq_length=gen_model.seq_length, representation=representation, target='mode', traintest='test', notes=notes)
    test_data_loader = DataLoader(test_dataset, 128, num_workers=4)

    vocab_size = test_dataset._vocab_size
    mode_num = test_dataset._mode_num

    print(f'Loaded dataset with {test_dataset._data_size} chants and a {representation} vocab size of {vocab_size}.')
        
    gen_model.to(device)

    # random.seed(1234)

    with torch.no_grad():
        ct = 0
        print('testing normal')
        test_normal_acc = 0
        for step, (test_batch_inputs, test_batch_next_targets, test_batch_mode_targets, test_batch_genre) in enumerate(test_data_loader):
            

            x = torch.stack(test_batch_inputs, dim=1).to(device)
            # x = num2hot(x, vocab_size, device)
            
            _, _, states = gen_model(x)
            h_state = states[0][1,:,:].cpu().numpy()
            inds = np.where((test_batch_genre == 22) | (test_batch_genre == 2))[0]
            ct += len(inds)
            test_normal_acc += (test_batch_mode_targets[inds].numpy() == mode_model.predict(h_state)[inds]).sum()


        # Initialize the dataset and data loader (note the +1)
        test_dataset = ChantDataset(seq_length=gen_model.seq_length, representation=representation, target='mode', traintest='test', notes=notes)
        test_data_loader = DataLoader(test_dataset, 128, num_workers=4)

        # test_loss = test_loss/ct
        test_normal_acc = test_normal_acc/ct

        # test_loss = 0
        test_shuffle_acc = 0
        ct = 0
        print('testing shuffle')
        for step, (test_batch_inputs, test_batch_next_targets, test_batch_mode_targets, test_batch_genre) in enumerate(test_data_loader):
            ct += 1

            for i in range(len(test_batch_inputs)):
                random.shuffle(test_batch_inputs[i])

            shuffle_x = torch.stack(test_batch_inputs, dim=1).to(device)
            # shuffle_x = num2hot(shuffle_x, vocab_size, device)
            

            _, _, states = gen_model(x)
            h_state = states[0][1,:,:].cpu().numpy()
            inds = np.where((test_batch_genre == 22) | (test_batch_genre == 2))[0]
            ct += len(inds)
            test_shuffle_acc += (test_batch_mode_targets[inds].numpy() == mode_model.predict(h_state)[inds]).sum()

        test_shuffle_acc = test_shuffle_acc/ct

        print(f'Shuffle acc: {test_shuffle_acc}, Normal acc: {test_normal_acc}')



 ################################################################################
 ################################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help="Name of the run")
    parser.add_argument('--device', type=str, default="cuda", help="Name of the run")

    config = parser.parse_args()

    # Train the model
    train(config)
