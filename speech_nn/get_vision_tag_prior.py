#!/usr/bin/env python

"""
Average tag probabilities on the training set are used for each test instance.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from os import path
import argparse
import cPickle as pickle
import numpy as np
import os
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "sigmoid_npz", type=str,
        help="Numpy archive with sigmoid activations from vision system"
        )
    parser.add_argument(
        "word_to_id_pkl", type=str,
        help="pickled dictionary giving ID to word mapping"
        )
    parser.add_argument(
        "subset", type=str, help="subset to get the baseline for",
        choices=["dev", "test"]
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    n_most_common = 1000

    word_to_id_dict_fn = args.word_to_id_pkl
    print "Reading:", word_to_id_dict_fn
    with open(word_to_id_dict_fn, "rb") as f:
        word_to_id = pickle.load(f)

    print "Filtering word IDs to {} most common".format(n_most_common)
    word_to_id_most_common = {}
    for word in word_to_id:
        word_id = word_to_id[word]
        if word_id < n_most_common:
            word_to_id_most_common[word] = word_id
    word_to_id = word_to_id_most_common

    # Train and subset utterances
    features_dict_fn = "data/mfcc_cmvn_dd_vad/{}.npz".format("train")
    print "Reading:", features_dict_fn
    features_dict = np.load(features_dict_fn)
    train_utterances = sorted(features_dict.keys())
    features_dict_fn = "data/mfcc_cmvn_dd_vad/{}.npz".format(args.subset)
    print "Reading:", features_dict_fn
    features_dict = np.load(features_dict_fn)
    subset_utterances = sorted(features_dict.keys())
    print "No. train utterances:", len(train_utterances)
    print "No. {} utterances: {}".format(args.subset, len(subset_utterances))

    # Vision features
    vision_npz = args.sigmoid_npz
    print "Reading:", vision_npz
    vision_features = np.load(vision_npz)

    output_dir = "models/vision_tag_prior_" + path.split(
        path.split(vision_npz)[0]
        )[1]
    if not path.isdir(output_dir):
        os.mkdir(output_dir)

    # Obtain average visual features
    n = 0.
    tag_prior_vector = np.zeros(n_most_common, dtype=np.float32)
    for utt_key in train_utterances:
        for image_key in vision_features:
            if image_key in utt_key:
                n += 1.
                tag_prior_vector += vision_features[image_key]
    tag_prior_vector = tag_prior_vector / n

    # Tag prior sigmoid_output_dict
    sigmoid_output_dict = {}
    for utt in sorted(subset_utterances):
        sigmoid_output_dict[utt] = tag_prior_vector
    sigmoid_output_dict_fn = path.join(output_dir, "sigmoid_output_dict." + args.subset + ".pkl")
    print "Writing:", sigmoid_output_dict_fn
    with open(sigmoid_output_dict_fn, "wb") as f:
        pickle.dump(sigmoid_output_dict, f, -1)

    # Other outputs required for valid model
    word_to_id_fn = path.join(output_dir, "word_to_id.pkl")
    print "Writing:", word_to_id_fn
    with open(word_to_id_fn, "wb") as f:
        pickle.dump(word_to_id, f, -1)
    options_dict_fn = path.join(output_dir, "options_dict.pkl")
    print "Writing:", options_dict_fn
    with open(options_dict_fn, "wb") as f:
        pickle.dump({"label_dict": "data/captions_content_dict.pkl"}, f, -1)


if __name__ == "__main__":
    main()



