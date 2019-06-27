from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

import torch
import matplotlib.pyplot as plt
from model import ModeModel
from chant_dataset import ChantDataset

from sklearn.decomposition import PCA

device = torch.device('cuda')

path = 'output/next_raw_pitch_embedding300_20_1024'   
model_save_string = path + '/model.pt'

gen_model = torch.load(model_save_string)

# Initialize the dataset and data loader (note the +1)
dataset = ChantDataset(seq_length=20, representation='raw', target='mode', traintest='train', notes='pitch')
voc = dataset._vocab

embedding = gen_model.embedding.weight.detach().cpu().numpy()

pca = PCA(n_components=2)

pca.fit(embedding)

embedding = pca.transform(embedding)

notes = ['9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', ')', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
mollenkruisen = ['y', 'Y', 'i', 'I', 'z', 'Z']

for i, n in enumerate(voc):
    if n in notes:
        print(i,n)
        plt.scatter(embedding[i,0], embedding[i,1], c='blue', marker=r"$ {} $".format(n))
        # plt.pause(0.2)
    elif n in mollenkruisen:
        print(i,n)
        plt.scatter(embedding[i,0], embedding[i,1], c='red', marker=r"$ {} $".format(n))
        # plt.pause(0.2)
    else:
        print(i,n)
        print(n,'n')
        if n in [' ', '\n', '\xa0']:
            plt.scatter(embedding[i,0], embedding[i,1], c='magenta')
        else:
            plt.scatter(embedding[i,0], embedding[i,1], c='magenta', marker=r"$ {} $".format(n))
        # plt.pause(0.2)


plt.show()