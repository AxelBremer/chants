import numpy as np
import pickle
import joblib
import argparse
import json
from collections import Counter

import matplotlib.pyplot as plt

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.corpora.import_corpus import import_corpus_from_path

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

    if 'embedding' in config.name:
        train_corpus = import_corpus_from_path('data/inputs/' + notes + '_' + str(seq_length) + '_' + representation + '_string_mode_corpus_train.txt', ['sen', 'labels'])
        test_corpus = import_corpus_from_path('data/inputs/' + notes + '_' + str(seq_length) + '_' + representation + '_string_mode_corpus_test.txt', ['sen', 'labels'])
        test_genre_corpus = import_corpus_from_path('data/inputs/' + notes + '_' + str(seq_length) + '_' + representation + '_string_genre_corpus_test.txt', ['sen', 'labels'])
    else:
        train_corpus = import_corpus_from_path('data/inputs/' + notes + '_' + str(seq_length) + '_' + representation + '_mode_corpus_train.txt', ['sen', 'labels'])
        test_corpus = import_corpus_from_path('data/inputs/' + notes + '_' + str(seq_length) + '_' + representation + '_mode_corpus_test.txt', ['sen', 'labels'])

    # hx_1_train = train_reader.read_activations((1,'hx'))
    # train_labels = np.zeros(int(hx_1_train.shape[0]/seq_length))
    # for i in range(len(train_labels)):
    #     train_labels[i] = train_corpus[i].labels[19]

    # train_scores = np.zeros(seq_length)
    # train_stds = np.zeros(seq_length)

    # for i in range(seq_length):
    #     activation_train = hx_1_train[i::seq_length]
    #     train_pred = classifier.predict(activation_train)
    #     train_scores[i] = (train_pred == train_labels).mean()


    # plt.plot(range(1,seq_length+1), train_scores, c='Red')

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
    most_common_genres = count.most_common(25)
    genre_vocab = [w.strip() for w in vocab_lines]

    resp_verse = genre_vocab.index('Responsory verse')
    antiphon = genre_vocab.index('Antiphon')

    genre_ind_dict = {}
    for genre in most_common_genres:
        print(genre_vocab[genre[0]], genre)
        genre_ind_dict[genre[0]] = np.where(np.array(test_genres)==genre[0])

    test_scores = np.zeros(seq_length)
    test_stds = np.zeros(seq_length)

    test_scores_genre = {}

    for genre in genre_ind_dict:
        test_scores_genre[genre] = np.zeros(seq_length)

    preds = []

    for i in range(seq_length):
        activation_test = hx_1_test[i::seq_length]
        test_pred = classifier.predict(activation_test)
        preds.append(test_pred)
        for genre in genre_ind_dict:
            tp = test_pred[genre_ind_dict[genre]]
            tl = test_labels[genre_ind_dict[genre]]
            test_scores_genre[genre][i] = (tp == tl).mean()
        # test_scores[i] = (test_pred == test_labels).mean()

    d = {'antiphon': test_scores_genre[antiphon].tolist(), 'resp_verse': test_scores_genre[resp_verse].tolist()}

    with open('output/' + config.name + '/scores.json', 'w') as f:
        json.dump(d, f)


    # plt.plot(range(1,seq_length+1), test_scores, label='All genres')
    for genre in genre_ind_dict:
        plt.plot(range(1,seq_length+1), test_scores_genre[genre], label=genre_vocab[genre])
    # plt.plot(range(1,21), test_scores+test_stds, c='Blue')
    # plt.plot(range(1,21), test_scores-test_stds, c='Blue')
    plt.legend()
    plt.xticks(list(range(1,seq_length+1)))
    plt.ylim(0,1)
    fig = plt.gcf()
    plt.show()
    ans = input('Do you want to save this figure? y/n\n')
    if ans == 'y':
        fig.savefig('output/' + config.name + '/eval_class.jpg')

    print('done')

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default="debug", help="Name of the run")
    config = parser.parse_args()

    # Train the model
    main(config)