#!/usr/bin/env python

"""
Train a bag-of-words MLP on top of MSCOCO VGG16 features.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from datetime import datetime
from os import path
import argparse
import cPickle as pickle
import hashlib
import numpy as np
import os
import random
import shutil
import sys
import tensorflow as tf

sys.path.append(path.join("..", "..", "src", "tflego"))

from tflego import blocks
from tflego import training
from tflego.blocks import TF_DTYPE, TF_ITYPE, NP_DTYPE


#-----------------------------------------------------------------------------#
#                           DEFAULT TRAINING OPTIONS                          #
#-----------------------------------------------------------------------------#

default_options_dict = {
    # "data_dir": "data/flickr30k", # "data/mscoco", # "data/temp", # 
    # "model_dir": "models/flickr30k/train_bow_mlp",
    # "train_list": "train",
    # "val_list": "dev",
    # "data_dir": "data/mscoco", # "data/mscoco", # "data/temp", # 
    # "model_dir": "models/mscoco/train_bow_mlp",
    # "train_list": "train",
    # "val_list": "val",
    "data_dir": "data/mscoco+flickr30k",
    "model_dir": "models/mscoco+flickr30k/train_bow_mlp",
    "train_list": "train",
    "val_list": "val",
    "content_words": True,  # if True, only train on content words
    "n_max_epochs": 100,  # 75
    "batch_size": 256,  # 256
    "ff_keep_prob": 0.75, # 0.75,
    "n_most_common": 1000,
    # "pos_weight": 100.0,  # 1.0 # if specified, the `weighted_cross_entropy_with_logits` loss is used
    "n_hiddens": [3072, 3072, 3072, 3072],
    # "optimizer": {
    #     "type": "sgd",
    #     "learning_rate": 0.001
    # },
    "optimizer": {
        "type": "adam",
        "learning_rate": 0.0001
    },
    "detect_sigmoid_threshold": 0.5,
    "train_bow_type": "single",  # "single", "average", "average_greg", "top_k"
    "rnd_seed": 0,
    "early_stopping": False, # True, # False,
    }


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])#, add_help=False)
    parser.add_argument(
        "--model_dir", type=str,
        help="if provided, this is the path where the model will be stored "
        "and where previous models would be searched for",
        default=None
        )
    return parser.parse_args()


def load_bow_labelled(features_dict, data_dir, subset_list, label_dict, n_bow, bow_type="single"):
    """
    Return the MSCOCO image matrices and bag-of-word label vectors.

    Parameters
    ----------
    bow_type : str
        How should the multiple captions be handled in constructing the
        bag-of-words vector: "single" assigns 1 to a word that occurs in any of
        the captions; "average" sums the word counts and then divides by the
        number of captions; "top_k" keeps only the top k most common words.
    """

    # Load data and shuffle
    subset_fn = path.join(data_dir, subset_list + ".txt")
    print "Reading:", subset_fn
    image_keys = []
    with open(subset_fn, "r") as f:
        for line in f:
            image_keys.append(line.strip())
    np.random.shuffle(image_keys)
    x = np.array([features_dict[i] for i in image_keys], dtype=NP_DTYPE)

    # Bag-of-word vectors
    print datetime.now()
    print "Getting bag-of-word vectors"
    bow_vectors = np.zeros((len(x), n_bow), dtype=NP_DTYPE)
    if bow_type == "single":
        for i_data, image_key in enumerate(image_keys):
            for i_caption in xrange(5):
                label_key = "{}_{}".format(image_key, i_caption)
                if not label_key in label_dict:
                    print "Warning: Missing label " + label_key
                else:
                    for i_word in label_dict[label_key]:
                        bow_vectors[i_data, i_word] = 1
    elif bow_type == "average":
        for i_data, image_key in enumerate(image_keys):
            for i_caption in xrange(5):
                label_key = "{}_{}".format(image_key, i_caption)
                if not label_key in label_dict:
                    print "Warning: Missing label " + label_key
                else:
                    for i_word in label_dict[label_key]:
                        bow_vectors[i_data, i_word] += 1
            bow_vectors[i_data, :] = bow_vectors[i_data, :]/5.
    elif bow_type == "top_k":
        k = 10
        for i_data, image_key in enumerate(image_keys):
            cur_count_vector = np.zeros(n_bow, dtype=NP_DTYPE)
            for i_caption in xrange(5):
                label_key = "{}_{}".format(image_key, i_caption)
                if not label_key in label_dict:
                    print "Warning: Missing label " + label_key
                else:
                    for i_word in label_dict[label_key]:
                        cur_count_vector[i_word] += 1
            top_k_indices = cur_count_vector.argsort()[-k:][::-1]
            bow_vectors[i_data, top_k_indices] = 1
    else:
        assert False

    return x, bow_vectors


#-----------------------------------------------------------------------------#
#                              TRAINING FUNCTIONS                             #
#-----------------------------------------------------------------------------#


def build_bow_mlp_from_options_dict(x, keep_prob, options_dict):
    mlp = blocks.build_feedforward(x, options_dict["n_hiddens"], keep_prob)
    with tf.variable_scope("ff_layer_final"):
        mlp = blocks.build_linear(mlp, options_dict["d_out"])
        print "Final linear layer shape:", mlp.get_shape().as_list()
    return mlp


def train_bow_mlp(options_dict=None, config=None, model_dir=None):
    """Train and save a bag-of-words MLP."""

    # PRELIMINARY

    assert (options_dict is not None) or (model_dir is not None)
    print datetime.now()

    # Output directory
    epoch_offset = 0
    load_model_fn = None
    if model_dir is None:
        hasher = hashlib.md5(repr(sorted(options_dict.items())).encode("ascii"))
        # hash_str = datetime.now().strftime("%y%m%d.%Hh%Mm%Ss") + "." + hasher.hexdigest()[:10]
        hash_str = hasher.hexdigest()[:10]
        model_dir = path.join(options_dict["model_dir"], hash_str)
        options_dict_fn = path.join(model_dir, "options_dict.pkl")
    else:
        # Start from previous model, if available
        options_dict_fn = path.join(model_dir, "options_dict.pkl")
        if path.isfile(options_dict_fn):
            print "Continuing from previous model"
            print "Reading:", options_dict_fn
            with open(options_dict_fn, "rb") as f:
                options_dict = pickle.load(f)
            epoch_offset = options_dict["n_epochs_complete"]
            print "Starting epoch:", epoch_offset            
            load_model_fn = path.join(model_dir, "model.n_epochs_{}.ckpt".format(epoch_offset))
    print "Model directory:", model_dir
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    print "Options:", options_dict

    # Model filename
    n_epochs_post_complete = epoch_offset + options_dict["n_max_epochs"]
    model_fn = path.join(model_dir, "model.n_epochs_{}.ckpt".format(n_epochs_post_complete))
    if options_dict["early_stopping"]:
        best_model_fn = path.join(model_dir, "model.best_val.ckpt")
    else:
        best_model_fn = None

    # Random seeds
    random.seed(options_dict["rnd_seed"])
    np.random.seed(options_dict["rnd_seed"])
    tf.set_random_seed(options_dict["rnd_seed"])


    # LOAD AND FORMAT DATA

    # Read word ID labels
    if options_dict["content_words"]:
        label_dict_fn = path.join(options_dict["data_dir"], "captions_word_ids_content_dict.pkl")
        word_to_id_fn = path.join(options_dict["data_dir"], "word_to_id_content.pkl")
    else:
        label_dict_fn = path.join(options_dict["data_dir"], "captions_word_ids_dict.pkl")
        word_to_id_fn = path.join(options_dict["data_dir"], "word_to_id.pkl")
    print "Reading:", label_dict_fn
    with open(label_dict_fn, "rb") as f:
        label_dict = pickle.load(f)
    word_to_id_dest_fn = path.join(model_dir, "word_to_id.pkl")
    print "Copying", word_to_id_fn
    shutil.copyfile(word_to_id_fn, word_to_id_dest_fn)

    # Filter out uncommon words (assume IDs sorted by count)
    print "Keeping most common words:", options_dict["n_most_common"]
    for image_key in sorted(label_dict):
        label_dict[image_key] = [i for i in label_dict[image_key] if i < options_dict["n_most_common"]]

    # Load image data
    npz_fn = path.join(options_dict["data_dir"], "fc7.npz")
    print "Reading:", npz_fn
    features_dict = np.load(npz_fn)
    train_x, train_y_bow = load_bow_labelled(
        features_dict, options_dict["data_dir"], options_dict["train_list"],
        label_dict, options_dict["n_most_common"],
        bow_type=options_dict["train_bow_type"]
        )
    dev_x, dev_y_bow = load_bow_labelled(
        features_dict, options_dict["data_dir"], options_dict["val_list"],
        label_dict, options_dict["n_most_common"]
        )
    print "Train items shape:", train_x.shape
    print "Dev items shape:", dev_x.shape

    # Dimensionalities
    d_in = train_x.shape[1]
    d_out = options_dict["n_most_common"]
    options_dict["d_in"] = d_in
    options_dict["d_out"] = d_out

    # Batch feed iterators
    class BatchFeedIterator(object):
        def __init__(self, x_mat, y_vec, keep_prob, shuffle_epoch=False):
            self._x_mat = x_mat
            self._y_vec = y_vec
            self._keep_prob = keep_prob
            self._shuffle_epoch = shuffle_epoch
        def __iter__(self):
            if self._shuffle_epoch:
                shuffle_indices = range(self._y_vec.shape[0])
                random.shuffle(shuffle_indices)
                self._x_mat = self._x_mat[shuffle_indices]
                self._y_vec = self._y_vec[shuffle_indices]
            n_batches = int(np.float(self._y_vec.shape[0] / options_dict["batch_size"]))
            for i_batch in xrange(n_batches):
                yield (
                    self._x_mat[
                        i_batch * options_dict["batch_size"]:(i_batch + 1) * options_dict["batch_size"]
                        ],
                    self._y_vec[
                        i_batch * options_dict["batch_size"]:(i_batch + 1) * options_dict["batch_size"]
                        ],
                    self._keep_prob
                    )
    train_batch_iterator = BatchFeedIterator(
        train_x, train_y_bow, options_dict["ff_keep_prob"], shuffle_epoch=True
        )
    val_batch_iterator = BatchFeedIterator(
        dev_x, dev_y_bow, 1.0, shuffle_epoch=False
        )


    # DEFINE MODEL

    print datetime.now()
    print "Building bag-of-words MLP"

    # Model
    x = tf.placeholder(TF_DTYPE, [None, d_in])
    y = tf.placeholder(TF_DTYPE, [None, d_out])
    keep_prob = tf.placeholder(TF_DTYPE)
    mlp = build_bow_mlp_from_options_dict(x, keep_prob, options_dict)

    # Training tensors
    if "pos_weight" in options_dict and options_dict["pos_weight"] != 1.:
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(mlp, y, options_dict["pos_weight"])
            )
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(mlp, y))
    if options_dict["optimizer"]["type"] == "sgd":
        optimizer_class = tf.train.GradientDescentOptimizer
    elif options_dict["optimizer"]["type"] == "momentum":
        optimizer_class = tf.train.MomentumOptimizer
    elif options_dict["optimizer"]["type"] == "adagrad":
        optimizer_class = tf.train.AdagradOptimizer
    elif options_dict["optimizer"]["type"] == "adadelta":
        optimizer_class = tf.train.AdadeltaOptimizer
    elif options_dict["optimizer"]["type"] == "adam":
        optimizer_class = tf.train.AdamOptimizer
    optimizer_kwargs = dict([i for i in options_dict["optimizer"].items() if i[0] != "type"])
    optimizer = optimizer_class(**optimizer_kwargs).minimize(loss)

    # Test tensors
    prediction = tf.cast(
        tf.greater_equal(tf.nn.sigmoid(mlp), options_dict["detect_sigmoid_threshold"]), TF_DTYPE
        )
    n_tp = tf.reduce_sum(prediction * y)
    n_pred = tf.reduce_sum(prediction)
    n_true = tf.reduce_sum(y)
    precision = n_tp/n_pred
    recall = n_tp/n_true
    fscore = 2.*precision*recall/(precision + recall)

    # TRAIN MODEL

    print(datetime.now())
    print "Training bag-of-words MLP"
    record_dict = training.train_fixed_epochs(
        options_dict["n_max_epochs"], optimizer, loss, train_batch_iterator,
        [x, y, keep_prob], [loss, precision, recall, -fscore],
        val_batch_iterator, load_model_fn=load_model_fn,
        save_model_fn=model_fn, config=config, epoch_offset=epoch_offset,
        save_best_val_model_fn=best_model_fn
        )

    # Save record
    record_dict_fn = path.join(model_dir, "record_dict.n_epochs_{}.pkl".format(n_epochs_post_complete))
    print "Writing:", record_dict_fn
    with open(record_dict_fn, "wb") as f:
        pickle.dump(record_dict, f, -1)

    # Save options_dict
    options_dict["n_epochs_complete"] = n_epochs_post_complete
    print("Writing: " + options_dict_fn)
    with open(options_dict_fn, "wb") as f:
        pickle.dump(options_dict, f, -1)

    print datetime.now()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Slurm options
    if "OMP_NUM_THREADS" in os.environ:
        num_threads = int(os.environ["OMP_NUM_THREADS"])
        config = tf.ConfigProto(intra_op_parallelism_threads=num_threads) #, log_device_placement=True)
    else:
        config = None

    # Set options
    options_dict = default_options_dict.copy()
    options_dict["script"] = "train_bow_mlp"

    # Train model
    train_bow_mlp(options_dict, config, model_dir=args.model_dir)


if __name__ == "__main__":
    main()

