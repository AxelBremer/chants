from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import argparse

import numpy as np
import json
import joblib
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

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

def get_token_from_output(output, temperature, method):
    if method == 'greedy':
        return torch.tensor([[torch.argmax(output).cpu().numpy().item()]], dtype=torch.long)
    else:
        probs = torch.softmax(output.squeeze()/temperature, dim=0)
    return torch.tensor([[torch.multinomial(probs,1).cpu().numpy().item()]])

def gen_text(model, states, dataset, config, gen_len, method):
    with torch.no_grad():
        device = torch.device(config.device)

        # Create empty sentence
        sentence = []

        # Create random first character
        first_char = torch.randint(low=0, high=dataset._vocab_size, size=(1,1)).to(device)

        # Add to sentence and convert to one hot
        sentence.append(first_char.cpu().numpy().item())
        # first_char = num2hot(first_char, dataset._vocab_size, device)

        # Get new character from model and add to sentence
        out, _, states = model(first_char, states)
        n_char = get_token_from_output(out, config.temperature, method).to(device)
        sentence.append(n_char.cpu().numpy().item())

        # Keep doing this for the required length
        for i in range(gen_len):
            # n_char = num2hot(n_char, dataset._vocab_size, device)

            out, _, states = model(n_char, states)

            n_char = get_token_from_output(out, config.temperature, method).to(device)

            sentence.append(n_char.cpu().numpy().item())

    return dataset.convert_to_string(sentence), sentence

def main(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    path = 'output/' + config.name    
    model_save_string = path + '/model.pt'
    mode_model_save_string = path + '/classifiers/hx_l1.joblib'

    if os.path.isfile(model_save_string):
        print('Loading model')
        model = torch.load(model_save_string, map_location=config.device)
        mode_model = joblib.load(mode_model_save_string)
    else:
        print('no gen model')

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
    dataset = ChantDataset(seq_length=model.seq_length, representation=representation, target=model.target, traintest='train', notes=notes)

    vocab_size = dataset._vocab_size

    right = 0
    total = 0

    for i in tqdm(range(2000)):
        with torch.no_grad():

            x, _, mode_target, genre = dataset[i]
            _, _, states = model(torch.Tensor([x[:12]]).to(device).long())
            pred = mode_model.predict(states[0][1,:,:].cpu().numpy())[0]
            if pred == mode_target:
                volp, inds = gen_text(model, states, dataset, config, 20, 'temp')
                _, _, states = model(torch.Tensor([inds]).to(device).long())
                pred = mode_model.predict(states[0][1,:,:].cpu().numpy())[0]
                total += 1
                right += pred == mode_target
                if pred == mode_target:
                    print(volp)
                    input('...')

    print('accuracy:', right/total)
    print(total)
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--device', type=str, default="cuda", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--name', type=str, default="debug", help="Name of the run")
    parser.add_argument('--temperature', type=float, default=0.5 , help="target [next] note or [mode]")

    config = parser.parse_args()

    # Train the model
    main(config)