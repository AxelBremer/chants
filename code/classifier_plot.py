import numpy as np
import pickle
import joblib
import argparse
import math
import json

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

with open('output/next_syllable_interval_embedding50_20_1024/scores.json', 'r') as f:
    syll_interval_d = json.load(f)

with open('output/next_syllable_pitch_embedding50_20_1024/scores.json', 'r') as f:
    syll_pitch_d = json.load(f)

with open('output/next_neume_interval_embedding50_20_1024/scores.json', 'r') as f:
    neume_interval_d = json.load(f)

with open('output/next_neume_pitch_embedding50_20_1024/scores.json', 'r') as f:
    neume_pitch_d = json.load(f)

with open('output/next_raw_interval_embedding20_20_1024/scores.json', 'r') as f:
    raw_interval_d = json.load(f)

with open('output/next_raw_pitch_embedding20_30_1024/scores.json', 'r') as f:
    raw_pitch_d = json.load(f)

pitchlist = [raw_pitch_d, neume_pitch_d, syll_pitch_d]
intervallist = [raw_interval_d, neume_interval_d, syll_interval_d]

colors = ['salmon', 'yellowgreen', 'steelblue']
names = ['Raw', 'Neume', 'Syllable']

for i,d in enumerate(pitchlist):
    antiphon_scores = d['antiphon']
    resp_scores = d['resp_verse']
    plt.plot(range(1,21), antiphon_scores[:20], label=names[i] + ' antiphon', linestyle='solid', c=colors[i])
    plt.plot(range(1,21), resp_scores[:20], label=names[i] + ' responsory verse', linestyle='dashdot', c=colors[i])
plt.ylim(0.2,1)
plt.xticks(range(1,21))
plt.xlabel('Timestep')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Pitch')
fig = plt.gcf()
fig.savefig('Pitch_classifiers.png')
plt.show()

for i,d in enumerate(intervallist):
    antiphon_scores = d['antiphon']
    resp_scores = d['resp_verse']
    plt.plot(range(1,21), antiphon_scores[:20], label=names[i] + ' antiphon', linestyle='solid', c=colors[i])
    plt.plot(range(1,21), resp_scores[:20], label=names[i] + ' responsory verse', linestyle='dashdot', c=colors[i])
plt.ylim(0.2,1)
plt.xticks(range(1,21))
plt.xlabel('Timestep')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Interval')
fig = plt.gcf()
fig.savefig('Interval_classifiers.png')
plt.show()