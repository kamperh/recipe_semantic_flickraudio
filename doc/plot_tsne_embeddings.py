#!/usr/bin/env python

"""
Plot t-SNE embeddings of two models.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from __future__ import division
from __future__ import print_function
from collections import Counter
from os import path
from sklearn import manifold
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import random

import plotting

KEYWORDS_FILTER = [
    "soccer", "football", "ball", "rides", "riding", "bike", "jumps", "air",
    "road", "street"
    ]
N_MAX_KEYWORDS = 50
TSNE_PERPLEXITY = 30


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def embeddings_from_pickle(pkl_fn):
    print("Reading:", pkl_fn)
    with open(pkl_fn, "rb") as f:
        features_dict = pickle.load(f)
    labels = []
    embeddings = []
    counter = Counter()
    for utt_key in features_dict:
        label = utt_key.split("_")[0]
        if label not in KEYWORDS_FILTER:
            continue
        if counter[label] > N_MAX_KEYWORDS:
            continue
        labels.append(label)
        counter[label] += 1
        embeddings.append(features_dict[utt_key])
    embeddings = np.array(embeddings)
    return embeddings, labels

def plot_labelled_2d_data(X, labels):
    classes = set(labels)
    for label in sorted(classes):
        indices = np.where(np.array(labels) == label)[0]
        plt.scatter(X[indices, 0], X[indices, 1], label=label)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    # SupervisedBoWCNN embeddings
    pkl_fn = path.join(
        "..", "speech_nn",
        "models/train_bow_cnn/12597afba4/"
        "sigmoid_final_feedforward_dict.dev_queries_all.pkl"
        )
    supervised_bow_cnn_embeds, labels = embeddings_from_pickle(pkl_fn)
    print("Embeddings shape:", supervised_bow_cnn_embeds.shape)
    print("Embeddings min:", np.min(supervised_bow_cnn_embeds))
    print("Embeddings max:", np.max(supervised_bow_cnn_embeds))
    print("Embeddings mean:", np.mean(supervised_bow_cnn_embeds))

    # Perform t-SNE
    tsne = manifold.TSNE(
        n_components=2, perplexity=TSNE_PERPLEXITY, init="random",
        random_state=0
        )
    supervised_bow_cnn_X_tsne = tsne.fit_transform(supervised_bow_cnn_embeds)

    # Plot t-SNE
    plotting.setup_plot()
    plt.rcParams["figure.figsize"]          = 5, 4.0
    plt.rcParams["figure.subplot.bottom"]   = 0.01
    plt.rcParams["figure.subplot.left"]     = 0.01
    plt.rcParams["figure.subplot.right"]    = 0.99
    plt.rcParams["figure.subplot.top"]      = 0.99
    plt.figure()
    plot_labelled_2d_data(supervised_bow_cnn_X_tsne, labels)
    plt.legend(loc="best", ncol=2)
    plt.xlim([-31, 34])
    plt.ylim([-45, 33])
    plt.yticks([])
    plt.xticks([])
    plt.savefig("train_bow_cnn_12597afba4_embeddings.pdf")

    print()

    # VisionSpeechCNN embeddings
    pkl_fn = path.join(
        "..", "speech_nn",
        "models/train_visionspeech_cnn/18ba6618ad/"
        "sigmoid_final_feedforward_dict.dev_queries_all.pkl"
        )
    vision_speech_cnn_embeds, labels = embeddings_from_pickle(pkl_fn)
    print("Embeddings shape:", vision_speech_cnn_embeds.shape)
    print("Embeddings min:", np.min(vision_speech_cnn_embeds))
    print("Embeddings max:", np.max(vision_speech_cnn_embeds))
    print("Embeddings mean:", np.mean(vision_speech_cnn_embeds))

    # Perform t-SNE
    tsne = manifold.TSNE(
        n_components=2, perplexity=TSNE_PERPLEXITY, init="random",
        random_state=0
        )
    vision_speech_cnn_X_tsne = tsne.fit_transform(vision_speech_cnn_embeds)

    # Plot t-SNE
    plotting.setup_plot()
    plt.rcParams["figure.figsize"]          = 5, 4.0
    plt.rcParams["figure.subplot.bottom"]   = 0.01
    plt.rcParams["figure.subplot.left"]     = 0.01
    plt.rcParams["figure.subplot.right"]    = 0.99
    plt.rcParams["figure.subplot.top"]      = 0.99
    plt.figure()
    plot_labelled_2d_data(vision_speech_cnn_X_tsne, labels) 
    plt.legend(loc="best", ncol=2)
    plt.xlim([-30, 28])
    plt.ylim([-28, 26])
    plt.yticks([])
    plt.xticks([])
    plt.savefig("train_visionspeech_cnn_18ba6618ad_embeddings.pdf")


if __name__ == "__main__":
    main()

