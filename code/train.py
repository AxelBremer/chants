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
import time
from datetime import datetime
import argparse

import numpy as np
import json
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ModeModel

################################################################################

def train(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(filename=config.txt_file, seq_length=config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=0)

    vocab_size = dataset.vocab_size
    
    save_string = config.txt_file.replace('.txt', '')
    text_save_string = save_string + '_' + str(config.seq_length) + '_greedy_generated_text.txt'
    model_save_string = save_string + '_' + str(config.seq_length) + '_model.pt'
    metric_save_string = save_string + '_' + str(config.seq_length) +  '_metrics.json'

    if not os.path.isfile(text_save_string):
        with open(text_save_string, 'w') as fp:
            fp.write('Generated text for ' + save_string)


    # Initialize the model that we are going to use
    if os.path.isfile(model_save_string):
        print('Loading model')
        print('No model found, creating one...')
        model = torch.load(model_save_string, map_location=config.device)
        print('Starting from %i steps in model' %(model.steps))
    else:
        model = TextGenerationModel(batch_size=config.batch_size, 
                                    seq_length=config.seq_length, 
                                    vocabulary_size=dataset.vocab_size, 
                                    lstm_num_hidden=config.lstm_num_hidden, 
                                    lstm_num_layers=config.lstm_num_layers, 
                                    device=config.device)
        
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    train_accuracy = np.zeros(int(config.train_steps)+1)

    out_steps = 0

    losses = []
    accs = []

    # Extra while loop to keep iterating over the dataset
    while out_steps < config.train_steps:
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            x = torch.stack(batch_inputs, dim=1).to(device)
            x = num2hot(x, vocab_size, device)

            y_target = torch.stack(batch_targets, dim=1).to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred.transpose(2,1), y_target)
            accuracy = get_accuracy(y_target, y_pred, config)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save number of steps in the model
            model.step()
            out_steps += 1

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Model Steps {:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        int(config.train_steps), model.steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                print('Sample at %i training steps with temp %f:' %(model.steps, config.temperature))
                t = gen_text(model, dataset, config, config.seq_length, 'prob')
                print(t)
                with open(text_save_string, 'a') as fp:
                    fp.write('\nSteps: ' + str(model.steps) + ' | ' + t)
                accs.append(accuracy)
                losses.append(loss.cpu().detach().item())
                torch.save(model, model_save_string)

            if out_steps == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    # Save metrics in file
    d = {'loss':losses, 'accuracy':accs}
    with open(metric_save_string, 'w') as fp:
        json.dump(d,fp)

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='Max to which to clip the norm of the gradients')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    # Misc params
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
