import numpy as np
import pickle
import joblib
import argparse
import math

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.corpora.import_corpus import import_corpus_from_path

from sklearn.manifold import TSNE

ct = 0

def show_tsne_plot(X, modes, show_legend=False, **kwargs):
    global ct
    colors = sns.color_palette(n_colors=4, palette='muted')
    labels = ['dorian', 'hypodorian', 'phrygian', 'hypophrygian',
            'lydian', 'hypolydian', 'mixolydian', 'hypomixolydian']

    for i in range(1, 9):
        is_mode = modes == i
        xs = X[is_mode, 0]
        ys = X[is_mode, 1]    
    
        # Plot properties
        props = dict(alpha=.5)
        props.update(kwargs)
        props['marker'] = 'o' if i % 2 == 0 else 'x'
        props['mew']    = 0   if i % 2 == 0 else .5
        props['ms']     = 1.5 if i % 2 == 0 else 2
        props['ms'] *= 5
        props['color'] = colors[int(math.ceil(i / 2)) - 1]
        props['label'] = f'{i}: {labels[i-1]}'           
    
        # Plot!
        plt.plot(xs, ys, lw=0, **props)

    plt.axis('off')
    plt.gca().get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().set_aspect('equal')
    if show_legend:
        leg = plt.legend(bbox_to_anchor=(1.0,1), loc="upper left", frameon=False)
        for lh in leg.legendHandles:
            lh._legmarker.set_alpha(1)
    fig = plt.gcf()
    plt.show()
    ans = input('Do you want to save this figure? y/n\n')
    if ans == 'y':
        fig.savefig('output/' + config.name + '/vis' + ct + '.jpg')
    ct += 1

def main(config):

    classifier = joblib.load('output/' + config.name + '/classifiers/hx_l1.joblib')
    train_reader = ActivationReader('output/' + config.name + '/activations/train')
    test_reader = ActivationReader('output/' + config.name + '/activations/test')

    if 'raw' in config.name:
        representation = 'raw'
    if 'neume' in config.name:
        representation = 'neume'
    if 'syllable' in config.name:
        representation = 'syllable'

    if 'pitch' in config.name:
        notes = 'pitch'
    if 'interval' in config.name:
        notes = 'interval'

    if '_20_' in config.name:
        seq_length = 20
    else:
        seq_length = 30

    # train_corpus = import_corpus_from_path('data/inputs/' + notes + '_20_' + representation + '_mode_corpus_train.txt', ['sen', 'labels'])
    test_corpus = import_corpus_from_path('data/inputs/' + notes + '_' + str(seq_length) + '_' + representation + '_mode_corpus_test.txt', ['sen', 'labels'])
    test_genre_corpus = import_corpus_from_path('data/inputs/' + notes + '_' + str(seq_length) + '_' + representation + '_string_genre_corpus_test.txt', ['sen', 'labels'])


    ##############################################################
    ##############################################################

    hx_1_test = test_reader.read_activations((1,'hx'))
    test_labels = np.zeros(int(hx_1_test.shape[0]/seq_length))
    test_genres = np.zeros(int(hx_1_test.shape[0]/seq_length))
    for i in range(len(test_labels)):
        test_labels[i] = test_corpus[i].labels[19]
        test_genres[i] = test_genre_corpus[i].labels[19]

    test_genres = test_genres.astype(int)
    test_labels = test_labels.astype(int)

    with open('data/inputs/' + notes + '_' + str(seq_length) + '_'  + representation + '_genre_vocab.txt', 'rb') as vf:
        vocab_lines = vf.readlines()
        vocab_lines = [line.decode('utf-8') for line in vocab_lines]

    count = Counter(test_genres)
    most_common_genres = count.most_common(3)
    genre_vocab = [w.strip() for w in vocab_lines]

    resp_verse = genre_vocab.index('Responsory verse')
    antiphon = genre_vocab.index('Antiphon')

    genre_ind_dict = {}
    for genre in most_common_genres:
        print(genre_vocab[genre[0]], genre)
        genre_ind_dict[genre[0]] = np.where(np.array(test_genres)==genre[0])

    ct = 1

    embs = {}

    for i in [0, 13]:
        activation_test = hx_1_test[i::20]
        activation_test = activation_test[genre_ind_dict[resp_verse]]
        x_emb = TSNE(n_components=2, verbose=2).fit_transform(activation_test)
        show_tsne_plot(x_emb, test_labels[genre_ind_dict[resp_verse]]+1)
        # plt.subplot(1,2,ct)
        # ct += 1
        # plt.tick_params(axis='x',          # changes apply to the x-axis
        #                 which='both',      # both major and minor ticks are affected
        #                 left=False,
        #                 bottom=False,      # ticks along the bottom edge are off
        #                 top=False,         # ticks along the top edge are off
        #                 labelbottom=False) # labels along the bottom edge are off
        # plt.tick_params(axis='y',          # changes apply to the x-axis
        #                 which='both',      # both major and minor ticks are affected
        #                 bottom=False,      # ticks along the bottom edge are off
        #                 top=False,         # ticks along the top edge are off
        #                 labelbottom=False) # labels along the bottom edge are off
        # plt.scatter(x_emb[:,0], x_emb[:,1], c=test_labels[genre_ind_dict[resp_verse]])

    # plt.savefig('output/' + config.name + '/visualization.jpg')
    # plt.show()

    # print('done')

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default="debug", help="Name of the run")
    config = parser.parse_args()

    # Train the model
    main(config)