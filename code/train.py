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
def load_model(model_save_string, vocab_size, mode_num):
    # Initialize the model that we are going to use
    if os.path.isfile(model_save_string) and (model_save_string != 'debug'):
        print('Loading model')
        model = torch.load(model_save_string, map_location=config.device)
        print('Starting from %i epochs in model' %(model.epochs))
    else:
        print('No model found, creating one...')
        model = ModeModel(batch_size=config.batch_size, 
                                    seq_length=config.seq_length, 
                                    vocab_size=vocab_size,
                                    mode_num=mode_num,
                                    target=config.target, 
                                    lstm_num_hidden=config.lstm_num_hidden, 
                                    lstm_num_layers=config.lstm_num_layers, 
                                    device=config.device)
    return model

def get_char_from_output(output, temperature, method):
    if method == 'greedy':
        return torch.tensor([[torch.argmax(output).cpu().numpy().item()]], dtype=torch.long)
    else:
        probs = torch.softmax(output.squeeze()/temperature, dim=0)
    return torch.tensor([[torch.multinomial(probs,1).cpu().numpy().item()]])


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
    # Many to many
    if config.target == 'next':
        return (y_pred.argmax(dim=2) == y_target).sum().cpu().numpy().item()/(config.seq_length * config.batch_size)
    # Many to one
    if config.target == 'mode':
        return (y_pred.argmax(dim=1) == y_target).sum().cpu().numpy().item()/(config.batch_size)


def train(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = ChantDataset(seq_length=config.seq_length, representation=config.representation, target=config.target, traintest='train', notes=config.notes)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=4)

    # Initialize the dataset and data loader (note the +1)
    test_dataset = ChantDataset(seq_length=config.seq_length, representation=config.representation, target=config.target, traintest='test', notes=config.notes)
    test_data_loader = DataLoader(test_dataset, config.batch_size, num_workers=4)

    vocab_size = dataset._vocab_size
    mode_num = dataset._mode_num

    path = 'output/' + config.name    
    model_save_string = path + '/'+ '_' + str(config.notes) + str(config.seq_length) + '_' + str(config.lstm_num_hidden) + '_' + str(config.lstm_num_layers) + '_model.pt'
    metric_save_string = path + '/'+ '_' + str(config.notes) + str(config.seq_length) + '_' + str(config.lstm_num_hidden) + '_' + str(config.lstm_num_layers) + '_metrics.json'

    os.makedirs(path, exist_ok=True)
    
    model = load_model(model_save_string, vocab_size, mode_num)
        
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    out_epochs = 0

    losses = []
    accs = []
    test_losses = []
    test_accs = []

    print('training')
    # Extra while loop to keep iterating over the dataset
    while out_epochs < config.train_epochs:
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            x = torch.stack(batch_inputs, dim=1).to(device)
            x = num2hot(x, vocab_size, device)

            if config.target == 'mode':
                # Many to many
                # y_target = batch_targets.unsqueeze(1).repeat(1,config.seq_length).to(device)
                # Many to one
                y_target = batch_targets.to(device)
                y_pred, _ = model(x)
                loss = criterion(y_pred, y_target).to(device)

            if config.target == 'next':
                y_target = torch.stack(batch_targets, dim=1).to(device)
                y_pred, _ = model(x)
                loss = criterion(y_pred.transpose(2,1), y_target).to(device)

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
            for step, (test_batch_inputs, test_batch_targets) in enumerate(test_data_loader):
                ct += 1
                x = torch.stack(test_batch_inputs, dim=1).to(device)
                x = num2hot(x, vocab_size, device)
                
                if config.target == 'mode':
                    # Many to many
                    # y_target = batch_targets.unsqueeze(1).repeat(1,config.seq_length).to(device)
                    # Many to one
                    y_target_test = test_batch_targets.to(device)
                    y_pred_test, _ = model(x)
                    test_loss += criterion(y_pred_test, y_target_test).item()

                if config.target == 'next':
                    y_target_test = torch.stack(test_batch_targets, dim=1).to(device)
                    y_pred_test, _ = model(x)
                    test_loss += criterion(y_pred_test.transpose(2,1), y_target_test).item()

                test_acc += get_accuracy(y_target_test, y_pred_test, config)

            test_loss = test_loss/ct
            test_acc = test_acc/ct

            train_loss = 0
            train_acc = 0
            ct = 0
            for step, (train_batch_inputs, train_batch_targets) in enumerate(data_loader):
                ct += 1
                x = torch.stack(train_batch_inputs, dim=1).to(device)
                x = num2hot(x, vocab_size, device)
                
                if config.target == 'mode':
                    # Many to many
                    # y_target = batch_targets.unsqueeze(1).repeat(1,config.seq_length).to(device)
                    # Many to one
                    y_target_train = train_batch_targets.to(device)
                    y_pred_train, _ = model(x)
                    train_loss += criterion(y_pred_train, y_target_train).item()

                if config.target == 'next':
                    y_target_train = torch.stack(train_batch_targets, dim=1).to(device)
                    y_pred_train, _ = model(x)
                    train_loss += criterion(y_pred_train.transpose(2,1), y_target_train).item()

                train_acc += get_accuracy(y_target_train, y_pred_train, config)

            train_loss = train_loss/ct
            train_acc = train_acc/ct

        # save number of epochs
        model.next_epoch()
        out_epochs += 1
            
        print("[{}] Total Epochs {:03d}, Current Epochs {:02d}/{:02d}, Batch Size = {},"
            " Accuracy = {:.3f}, Loss = {:.3f}, Test Acc = {:.3f}, Test Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                model.epochs, out_epochs, int(config.train_epochs), config.batch_size,
                train_acc, train_loss, test_acc, test_loss
        ))

        accs.append(train_acc)
        losses.append(train_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        # Save model
        torch.save(model, model_save_string)
        # Save metrics in file
        d = {'loss':losses, 'accuracy':accs, 'test_loss':test_losses, 'test_accuracy':test_accs}
        with open(metric_save_string, 'w') as fp:
            json.dump(d,fp)
        print('Saved model and metrics')
    print('Done training.')


 ################################################################################
 ################################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=256, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--train_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--max_norm', type=float, default=5.0, help='Max to which to clip the norm of the gradients')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--device', type=str, default="cuda", help="Training device 'cpu' or 'cuda:0'")

    # Misc params
    parser.add_argument('--save_every', type=int, default=200, help='How often to save model and metrics')
    parser.add_argument('--name', type=str, default="debug", help="Name of the run")

    parser.add_argument('--representation', type=str, required=True, help="representation of the volpiano")
    parser.add_argument('--notes', type=str, default='interval', help="pitch or interval")

    parser.add_argument('--target', type=str, default='next', help="target [next] note or [mode]")

    config = parser.parse_args()

    # Train the model
    train(config)
