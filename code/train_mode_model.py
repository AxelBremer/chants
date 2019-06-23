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
from datetime import datetime
import argparse

import numpy as np
import json
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from model import ModeModel
from chant_dataset import ChantDataset

################################################################################
def load_model(gen_model, mode_model_save_string, metric_save_string, vocab_size, mode_num):
    # Initialize the model that we are going to use
    if os.path.isfile(mode_model_save_string):
        mode_model = torch.load(mode_model_save_string, map_location=config.device)
        print('Starting from %i epochs in model' %(mode_model.epochs))
        with open(metric_save_string, 'r') as fp:
            mode_metrics = json.load(fp)
        best_loss = min(mode_metrics['test_loss'])
    else:
        print('No model found, creating one...')
        mode_model = ModeModel(batch_size=config.batch_size, 
                                    seq_length=gen_model.seq_length, 
                                    vocab_size=vocab_size,
                                    mode_num=mode_num,
                                    target='mode',
                                    lstm_num_hidden=gen_model.lstm_num_hidden, 
                                    lstm_num_layers=gen_model.lstm_num_layers, 
                                    device=config.device)
        mode_metrics = d = {'loss':[], 'accuracy':[], 'test_loss':[], 'test_accuracy':[]}
        best_loss = 10000
        
    return mode_model, mode_metrics, best_loss

def num2hot(batch, vocab_size, dim, device):
    # Get the shape of the input and add the vocabulary size in a new dimension
    shape = list(batch.shape)
    shape = shape + [vocab_size]

    # Create the output tensor and use it as index to place a one in the new tensor
    y_out = torch.zeros(shape).to(device)
    batch = batch.unsqueeze(-1).long()

    y_out.scatter_(dim, batch, torch.tensor(1).to(device))

    return y_out


def get_accuracy(y_target, y_pred, config):
    return (y_pred.argmax(dim=1) == y_target).sum().cpu().numpy().item()/(config.batch_size)


def train(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    path = 'output/' + config.name    
    model_save_string = path + '/model.pt'
    mode_model_save_string = path + '/mode_model.pt'
    weight_save_string = path + '/mode_weights.pt'
    metric_save_string = path + '/mode_metrics.json'

    if os.path.isfile(model_save_string):
        print('Loading model')
        gen_model = torch.load(model_save_string, map_location=config.device)
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

    # Initialize the dataset and data loader (note the +1)
    dataset = ChantDataset(seq_length=gen_model.seq_length, representation=representation, target='mode', traintest='train', notes=notes)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=4)

    # Initialize the dataset and data loader (note the +1)
    test_dataset = ChantDataset(seq_length=gen_model.seq_length, representation=representation, target='mode', traintest='test', notes=notes)
    test_data_loader = DataLoader(test_dataset, config.batch_size, num_workers=4)

    vocab_size = dataset._vocab_size
    mode_num = dataset._mode_num

    print(f'Loaded dataset with {dataset._data_size} chants and a {representation} vocab size of {vocab_size}.')
    
    mode_model, metrics, best_loss = load_model(gen_model, mode_model_save_string, metric_save_string, vocab_size, mode_num)
        
    mode_model.to(device)
    gen_model.to(device)

    # copy lstm weights
    mode_model.lstm.load_state_dict(gen_model.lstm.state_dict())

    for param in mode_model.lstm.parameters():
        param.requires_grad= False

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(mode_model.parameters(), lr=config.learning_rate)

    out_epochs = 0

    print('training')
    # Extra while loop to keep iterating over the dataset
    while out_epochs < config.train_epochs:
        for step, (batch_inputs, batch_next_targets, batch_mode_targets) in enumerate(data_loader):
            x = torch.stack(batch_inputs, dim=1).to(device)
            x = num2hot(x, vocab_size, 2, device)

            # Many to many
            # y_target = batch_targets.unsqueeze(1).repeat(1,config.seq_length).to(device)
            # Many to one
            y_target = batch_mode_targets.to(device)
            _, y_mode_pred, _ = mode_model(x)
            loss = criterion(y_mode_pred, y_target).to(device)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        print('Calculating metrics')
        with torch.no_grad():
            # train_loss = running_loss/ct
            # train_acc = running_acc/ct

            test_loss = 0
            test_acc = 0
            ct = 0
            for step, (test_batch_inputs, test_batch_next_targets, test_batch_mode_targets) in enumerate(test_data_loader):
                ct += 1
                x = torch.stack(test_batch_inputs, dim=1).to(device)
                x = num2hot(x, vocab_size, 2, device)
                

                # Many to many
                # y_target = batch_targets.unsqueeze(1).repeat(1,config.seq_length).to(device)
                # Many to one
                y_target_test = test_batch_mode_targets.to(device)
                _, y_pred_test, _ = mode_model(x)
                test_loss += criterion(y_pred_test, y_target_test).item()
                test_acc += get_accuracy(y_target_test, y_pred_test, config)


            test_loss = test_loss/ct
            test_acc = test_acc/ct


            train_loss = 0
            train_acc = 0
            ct = 0
            for step, (train_batch_inputs, train_batch_next_targets, train_batch_mode_targets) in enumerate(data_loader):
                ct += 1
                x = torch.stack(train_batch_inputs, dim=1).to(device)
                x = num2hot(x, vocab_size, 2, device)
                

                # Many to many
                # y_target = batch_targets.unsqueeze(1).repeat(1,config.seq_length).to(device)
                # Many to one
                y_target_train = train_batch_mode_targets.to(device)
                _, y_pred_train, _ = mode_model(x)
                train_loss += criterion(y_pred_train, y_target_train).item()
                train_acc += get_accuracy(y_target_train, y_pred_train, config)

            train_loss = train_loss/ct
            train_acc = train_acc/ct

        # save number of epochs
        mode_model.next_epoch()
        out_epochs += 1
            
        print("[{}] Total Epochs {:03d}, Current Epochs {:02d}/{:02d}, Batch Size = {},"
            " Accuracy = {:.3f}, Loss = {:.3f}, Test Acc = {:.3f}, Test Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                mode_model.epochs, out_epochs, int(config.train_epochs), config.batch_size,
                train_acc, train_loss, test_acc, test_loss
        ))

        metrics['accuracy'].append(train_acc)
        metrics['loss'].append(train_loss)
        metrics['test_accuracy'].append(test_acc)
        metrics['test_loss'].append(test_loss)
        # Save mode_model
        if metrics['test_loss'][-1] < best_loss:
            print('best loss, saving mode_model')
            best_loss = metrics['test_loss'][-1]
            torch.save(mode_model, mode_model_save_string)
            torch.save(mode_model.state_dict(), weight_save_string)
        # Save metrics in file
        with open(metric_save_string, 'w') as fp:
            json.dump(metrics,fp)
        print('Saved metrics')
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Training params
    parser.add_argument('--batch_size', type=int, default=256, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--train_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--max_norm', type=float, default=5.0, help='Max to which to clip the norm of the gradients')

    parser.add_argument('--device', type=str, default="cuda", help="Training device 'cpu' or 'cuda:0'")

    # Misc params
    parser.add_argument('--save_every', type=int, default=200, help='How often to save model and metrics')
    parser.add_argument('--name', type=str, required=True, help="Name of the run")

    config = parser.parse_args()

    # Train the model
    train(config)
