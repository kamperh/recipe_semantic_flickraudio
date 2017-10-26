#!/usr/bin/env python

"""
Evaluate the given model on semantic keyword spotting.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from __future__ import print_function
from collections import Counter
from os import path
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import spearmanr
import argparse
import cPickle as pickle
import numpy as np
import sklearn.metrics as metrics
import sys

keywords_fn = path.join("..", "data", "keywords.8.txt")
exact_keywords_dict_fn = path.join(
    "..", "data", "06-16-23h59_exact_matches_dict.pkl"
    )
semkeywords_dict_fn = path.join(
    "..", "data", "06-16-23h59_semantic_labels_dict.pkl"
    )
batch_dict_fn = path.join("..", "data", "06-16-23h59_batch_dict.pkl")


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "--analyze",
        help="print an analysis of the evaluation output for each keyword",
        action="store_true"
        )
    parser.add_argument("model_dir", type=str, help="model directory")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                             EVALUATION FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def calculate_eer(y_true, y_score):
    # https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer


def eval_keyword_spotting(prediction_dict, true_dict, keyword_counts,
        analyze=False, captions_dict=None):
    """Return P@10, P@N and EER."""
    
    keywords = sorted(keyword_counts)
    utt_keys = sorted(prediction_dict)

    # Get sigmoid matrix for keywords
    prediction_vectors = np.zeros((len(utt_keys), len(keywords)))
    for i_utt, utt in enumerate(utt_keys):
        for i_keyword, keyword in enumerate(keywords):
            prediction_vectors[i_utt, i_keyword] = prediction_dict[utt][
                keyword
                ]

    # # Temp
    # outputs = []

    # Keyword spotting evaluation
    p_at_10 = []
    p_at_n = []
    eer = []
    for i_keyword, keyword in enumerate(keywords):

        # Rank
        rank_order = prediction_vectors[:, i_keyword].argsort()[::-1]
        utt_order = [utt_keys[i] for i in rank_order]

        # EER
        y_true = []
        for utt in utt_order:
            if keyword in true_dict[utt]:
                y_true.append(1)
            else:
                y_true.append(0)
        y_score = prediction_vectors[:, i_keyword][rank_order]
        cur_eer = calculate_eer(y_true, y_score)
        eer.append(cur_eer)

        # P@10
        cur_p_at_10 = float(sum(y_true[:10]))/10.
        p_at_10.append(cur_p_at_10)

        # P@N
        cur_p_at_n = float(sum(y_true[:sum(y_true)]))/sum(y_true)
        p_at_n.append(cur_p_at_n)

        if analyze:
            print("-"*79)
            print("Keyword:", keyword)
            print("Current P@10: {:.4f}".format(cur_p_at_10))
            print("Current P@N: {:.4f}".format(cur_p_at_n))
            print("Current EER: {:.4f}".format(cur_eer))
            print("Top 10 utterances:")
            for i_utt, utt in enumerate(utt_order[:10]):
                print(
                    "{}: {}".format(utt, " ".join(captions_dict[utt])), end=""
                    )
                if y_true[i_utt] == 0:
                    print(" *")
                else:
                    print()

                # # Temp
                # if i_utt == 0:
                #     if y_true[i_utt] == 0:
                #         outputs.append(keyword + " & " + " ".join(captions_dict[utt]) + " $*$ \\\\")
                #     else:
                #         outputs.append(keyword + " & " + " ".join(captions_dict[utt]) + " \\\\")

    if analyze:
        print("-"*79)
        print()

    # # Temp
    # for bla in outputs:
    #     print(bla)

    # Average
    p_at_10 = np.mean(p_at_10)
    p_at_n = np.mean(p_at_n)
    eer = np.mean(eer)

    return p_at_10, p_at_n, eer


def get_average_precision(prediction_dict, true_dict, keywords,
        show_plot=False):

    keywords = sorted(keywords)
    utt_keys = sorted(prediction_dict)

    # Prediction and true vectors
    prediction_vectors = np.zeros((len(utt_keys), len(keywords)))
    true_vectors = np.zeros((len(utt_keys), len(keywords)))
    for i_utt, utt in enumerate(utt_keys):
        for i_keyword, keyword in enumerate(keywords):
            prediction_vectors[i_utt, i_keyword] = prediction_dict[utt][
                keyword
                ]
            if keyword in true_dict[utt]:
                true_vectors[i_utt, i_keyword] = 1
        
    # Average precision
    ap = metrics.average_precision_score(
        true_vectors, prediction_vectors, average="micro"
        )
    
    if show_plot:
        import matplotlib.pyplot as plt
        precisions, recalls, _ = metrics.precision_recall_curve(
            true_vectors.ravel(), prediction_vectors.ravel()
            )
        plt.plot(recalls, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()
    
    return ap


def get_mean_average_precision(prediction_dict, true_dict, keywords,
        show_plot=False):

    keywords = sorted(keywords)
    utt_keys = sorted(prediction_dict)

    average_precisions = []
    for keyword in keywords:

        # Prediction and true vectors
        prediction_vectors = np.zeros(len(utt_keys))
        true_vectors = np.zeros(len(utt_keys))
        for i_utt, utt in enumerate(utt_keys):
            # The addition of random noise helps if there are only one
            # threshold, in which case the AP is always very close to 0.5,
            # independent of well the model does.
            prediction_vectors[i_utt] = (
                prediction_dict[utt][keyword] + np.random.normal(0, 0.0001, 1)
                )
            if keyword in true_dict[utt]:
                true_vectors[i_utt] = 1
            # print(true_vectors[i_utt], prediction_vectors[i_utt])

        # Average precision
        ap = metrics.average_precision_score(
            true_vectors, prediction_vectors, average="micro"
            )
        average_precisions.append(ap)

        # print(keyword, ap)
        # import matplotlib.pyplot as plt
        # precisions, recalls, _ = metrics.precision_recall_curve(
        #     true_vectors.ravel(), prediction_vectors.ravel()
        #     )
        # plt.plot(recalls, precisions)
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.show()
        # assert False
    
    # Mean average precision
    mAP = np.mean(average_precisions)
    
    return mAP


def eval_semkeyword_exact_sem_counts(prediction_dict, keywords,
        exact_label_dict, sem_label_dict):
    """
    Return a dict with elements `(n_total, n_exact, n_sem)` for each keyword.
    """

    keywords = sorted(keywords)
    utt_keys = sorted(prediction_dict)

    # Get sigmoid matrix for keywords
    prediction_vectors = np.zeros((len(utt_keys), len(keywords)))
    for i_utt, utt in enumerate(utt_keys):
        for i_keyword, keyword in enumerate(keywords):
            prediction_vectors[i_utt, i_keyword] = prediction_dict[utt][keyword]

    precision_counts_dict = {}  # [n_total, n_exact, n_sem]
    n_true_total_sem = 0
    n_true_total_exact = 0
    for i_keyword, keyword in enumerate(keywords):

        # Rank
        rank_order = prediction_vectors[:, i_keyword].argsort()[::-1]
        utt_order = [utt_keys[i] for i in rank_order]

        y_true = []
        y_exact = []
        y_sem = []
        for utt in utt_order:
            if keyword in sem_label_dict[utt]:
                y_true.append(1)
            else:
                y_true.append(0)
            if keyword in exact_label_dict[utt]:
                y_exact.append(1)
                y_sem.append(0)
            elif keyword in sem_label_dict[utt]:
                y_exact.append(0)
                y_sem.append(1)
            else:
                y_exact.append(0)
                y_sem.append(0)

        n_total = sum(y_true)
        n_exact = sum(y_exact[:sum(y_true)])
        n_sem = sum(y_sem[:sum(y_true)])

        n_true_total_sem += sum(y_sem)
        n_true_total_exact += sum(y_exact)

        precision_counts_dict[keyword] = (n_total, n_exact, n_sem)

    return precision_counts_dict, n_true_total_sem, n_true_total_exact


def get_spearmanr(prediction_dict, rating_dict, keywords):

    keywords = sorted(keywords)
    utt_keys = sorted(prediction_dict)

    prediction_vector = np.zeros(len(utt_keys)*len(keywords))
    rating_vector = np.zeros(len(utt_keys)*len(keywords))

    i_var = 0
    for utt in utt_keys:
        for keyword in keywords:
            prediction_vector[i_var] = prediction_dict[utt][keyword]
            if keyword in rating_dict[utt]:
                rating_vector[i_var] = rating_dict[utt][keyword]
            else:
                rating_vector[i_var] = 0
            i_var += 1

    return spearmanr(prediction_vector, rating_vector)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    subset = "test"

    # Load transcriptions
    if args.analyze:
        from get_captions import captions_fn, get_captions_dict
        captions_dict = get_captions_dict(captions_fn)
        tmp_dict = {}
        for utt_key in captions_dict:
            tmp_dict[utt_key[4:]] = captions_dict[utt_key]
        captions_dict = tmp_dict
    else:
        captions_dict = None

    # Read keywords
    # keywords_fn = path.join("..", "4.extra_keywords", "data", "keywords.8.txt")
    print("Reading:", keywords_fn)
    keywords = []
    with open(keywords_fn) as f:
        for line in f:
            keywords.append(line.strip())
    keywords = sorted(keywords)
    print("No. keywords:", len(keywords))

    # Read true labels
    # semkeywords_dict_fn = path.join(
    #     "..", "4.extra_keywords", "runs", "06-16_combined", "processed_23h59",
    #     "semantic_labels_dict.pkl"
    #     )
    print("Reading", semkeywords_dict_fn)
    with open(semkeywords_dict_fn, "rb") as f:
        semkeywords_dict = pickle.load(f)
    # exact_keywords_dict_fn = path.join(
    #     "..", "4.extra_keywords", "runs", "06-16_combined", "processed_23h59",
    #     "exact_matches_dict.pkl"
    #     )
    print("Reading", exact_keywords_dict_fn)
    with open(exact_keywords_dict_fn, "rb") as f:
        exact_keywords_dict = pickle.load(f)
    # batch_dict_fn = path.join(
    #     "..", "4.extra_keywords", "runs", "06-16_combined", "processed_23h59",
    #     "batch_dict.pkl"
    #     )
    print("Reading", batch_dict_fn)
    with open(batch_dict_fn, "rb") as f:
        batch_dict = pickle.load(f)

    # Obtain the exact and semantic keyword counts for each keyword
    exact_keywords_tokens = []
    semkeywords_tokens = []
    for utt in sorted(semkeywords_dict):
        exact_keywords_tokens.extend(exact_keywords_dict[utt])
        semkeywords_tokens.extend(semkeywords_dict[utt])
    counter = Counter(exact_keywords_tokens)
    exact_keywords_counts = dict(
        [(i, counter[i]) for i in counter if i in keywords]
        )
    counter = Counter(semkeywords_tokens)
    semkeywords_counts = dict(
        [(i, counter[i]) for i in counter if i in keywords]
        )
    print("No. utterances:", len(semkeywords_dict))

    # Load mapping dict
    word_to_id_fn = path.join(args.model_dir, "word_to_id.pkl")
    print("Reading: " + word_to_id_fn)
    with open(word_to_id_fn, "rb") as f:
        word_to_id = pickle.load(f)
    id_to_word = dict([(i[1], i[0]) for i in word_to_id.iteritems()])

    # Read sigmoid output
    sigmoid_output_dict_fn = path.join(
        args.model_dir, "sigmoid_output_dict." + subset + ".pkl"
        )
    print("Reading: " + sigmoid_output_dict_fn)
    with open(sigmoid_output_dict_fn, "rb") as f:
        sigmoid_output_dict = pickle.load(f)

    # Get similarity dict
    similarity_dict = {}
    for utt_key in sigmoid_output_dict:
        tmp_utt_key = utt_key[4:]
        if tmp_utt_key in semkeywords_dict:
            similarity_dict[tmp_utt_key] = {}
            for keyword in sorted(keywords):
                similarity_dict[tmp_utt_key][keyword] = sigmoid_output_dict[
                    utt_key
                    ][word_to_id[keyword]]

    # # Read similarity
    # similarity_dict_fn = path.join(args.model_dir, "similarity_dict.pkl")
    # print("Reading:", similarity_dict_fn)
    # with open(similarity_dict_fn, "rb") as f:
    #     similarity_dict = pickle.load(f)

    # Evaluate exact keyword spotting
    p_at_10, p_at_n, eer = eval_keyword_spotting(
        similarity_dict, exact_keywords_dict, exact_keywords_counts
        )
    average_precision = get_average_precision(
        similarity_dict, exact_keywords_dict, keywords
        )
    mean_average_precision = get_mean_average_precision(
        similarity_dict, exact_keywords_dict, keywords
        )

    print()
    print("-"*79)
    print("Exact keyword spotting:")
    print("Average P@10: {:.4f}".format(p_at_10))
    print("Average P@N: {:.4f}".format(p_at_n))
    print("Average EER: {:.4f}".format(eer))
    print("Average precision: {:.4f}".format(average_precision))
    print("Mean average precision: {:.4f}".format(mean_average_precision))

    # Evaluate semantic keyword spotting
    p_at_10, p_at_n, eer = eval_keyword_spotting(
        similarity_dict, semkeywords_dict, semkeywords_counts, args.analyze,
        captions_dict
        )
    average_precision = get_average_precision(
        similarity_dict, semkeywords_dict, keywords
        )
    mean_average_precision = get_mean_average_precision(
        similarity_dict, semkeywords_dict, keywords
        )
    spearmans_rho = get_spearmanr(
        similarity_dict, batch_dict, keywords
        )

    print("-"*79)
    print("Semantic keyword spotting:")
    print("Average P@10: {:.4f}".format(p_at_10))
    print("Average P@N: {:.4f}".format(p_at_n))
    print("Average EER: {:.4f}".format(eer))
    print("Average precision: {:.4f}".format(average_precision))
    print("Mean average precision: {:.4f}".format(mean_average_precision))
    print("Spearman's rho: {:.4f}".format(spearmans_rho[0]))

    # Breakdown separating out exact and semantic matches
    keyword_precision_counts, n_true_total_sem, n_true_total_exact = (
        eval_semkeyword_exact_sem_counts(similarity_dict, keywords,
        exact_keywords_dict, semkeywords_dict )
        )
    p_at_n_accum = 0
    n_total_accum = 0
    n_exact_accum = 0
    n_sem_accum = 0
    for keyword in sorted(keyword_precision_counts):
        (n_total, n_exact, n_sem) = keyword_precision_counts[keyword] 
        p_at_n_accum += float(n_exact + n_sem) / n_total
        n_total_accum += n_total
        n_exact_accum += n_exact
        n_sem_accum += n_sem
    
    print("-"*79)
    print("Breakdown of exact and semantic matches")
    print(
        "Average P@N: {:.4f}".format(p_at_n_accum /
        len(keyword_precision_counts))
        )
    print(
        "Average P@N* overall: {:.4f}".format(float(n_exact_accum +
        n_sem_accum) / n_total_accum)
        )
    print(
        "Average P@N* exact: {:.4f}".format(float(n_exact_accum ) /
        n_total_accum)
        )
    print(
        "Average P@N* semantic: {:.4f}".format(float(n_sem_accum) /
        n_total_accum)
        )
    # print(n_total_accum, n_true_total_sem, n_true_total_exact)
    print("-"*79)


if __name__ == "__main__":
    main()
