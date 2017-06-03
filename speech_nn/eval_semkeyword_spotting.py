#!/usr/bin/env python

"""
Evaluate a given model in semantic keyword spotting on labelled test data.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from collections import Counter
from os import path
import argparse
import cPickle as pickle
import numpy as np
import sys

from eval_keyword_spotting import eval_keyword_spotting

keywords_fn = path.join("..", "data", "keywords.6.txt")
exact_keywords_dict_fn = path.join("..", "data", "05-31-13h45_exact_matches_dict.pkl")
semkeywords_dict_fn = path.join("..", "data", "05-31-13h45_semantic_labels_dict.pkl")



#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("model_dir", type=str, help="model directory")
    # parser.add_argument(
    #     "subset", type=str, help="subset to perform evaluation on", choices=["train", "dev", "test"]
    #     )
    parser.add_argument(
        "--analyze", help="print an analysis of the evaluation output for each utterance",
        action="store_true"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                             EVALUATION FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def eval_semkeyword_exact_sem_counts(sigmoid_dict, word_to_id, keyword_list,
        exact_label_dict, sem_label_dict):
    """
    Return a dict with elements `(n_total, n_exact, n_sem)` for each keyword.
    """

    keywords = sorted(keyword_list)
    utterances = sorted(sigmoid_dict)
    keyword_ids = [word_to_id[w] for w in keywords]

    # Get sigmoid matrix for keywords
    keyword_sigmoid_mat = np.zeros((len(utterances), len(keywords)))
    for i_utt, utt in enumerate(utterances):
        keyword_sigmoid_mat[i_utt, :] = sigmoid_dict[utt][keyword_ids]

    precision_counts_dict = {}  # [n_total, n_exact, n_sem]
    for i_keyword, keyword in enumerate(keywords):

        # Rank
        rank_order = keyword_sigmoid_mat[:, i_keyword].argsort()[::-1]
        utt_order = [utterances[i] for i in rank_order]

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

        precision_counts_dict[keyword] = (n_total, n_exact, n_sem)

    return precision_counts_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    subset = "test"

    # Read keywords
    print("Reading: " + keywords_fn)
    with open(keywords_fn, "r") as f:
        keywords = []
        for line in f:
            keywords.append(line.strip())
    print("Keywords: " + str(keywords))

    print("Reading: " + exact_keywords_dict_fn)
    with open(exact_keywords_dict_fn, "rb") as f:
        exact_keywords_dict = pickle.load(f)

    print("Reading: " + semkeywords_dict_fn)
    with open(semkeywords_dict_fn, "rb") as f:
        semkeywords_dict = pickle.load(f)

    # Load mapping dict
    word_to_id_fn = path.join(args.model_dir, "word_to_id.pkl")
    print("Reading: " + word_to_id_fn)
    with open(word_to_id_fn, "rb") as f:
        word_to_id = pickle.load(f)
    id_to_word = dict([(i[1], i[0]) for i in word_to_id.iteritems()])

    # Read sigmoid output
    sigmoid_output_dict_fn = path.join(args.model_dir, "sigmoid_output_dict." + subset + ".pkl")
    print("Reading: " + sigmoid_output_dict_fn)
    with open(sigmoid_output_dict_fn, "rb") as f:
        sigmoid_output_dict = pickle.load(f)

    # Remove the speaker info from the keys and get the subset
    sigmoid_output_dict_subset = {}
    for utt_label in sigmoid_output_dict:
        tmp_utt_label = utt_label[4:]
        if tmp_utt_label in semkeywords_dict:
            sigmoid_output_dict_subset[tmp_utt_label] = sigmoid_output_dict[utt_label]

    # Obtain the exact and semantic keyword counts for each keyword
    exact_keywords_tokens = []
    semkeywords_tokens = []
    for utt in sorted(semkeywords_dict):
        exact_keywords_tokens.extend(exact_keywords_dict[utt])
        semkeywords_tokens.extend(semkeywords_dict[utt])
    counter = Counter(exact_keywords_tokens)
    exact_keywords_counts = dict([(i, counter[i]) for i in counter if i in keywords])
    counter = Counter(semkeywords_tokens)
    semkeywords_counts = dict([(i, counter[i]) for i in counter if i in keywords])
    print("No. utterances: " + str(len(semkeywords_dict)))
    # print Counter(semkeywords_counts).most_common()
    # print Counter(exact_keywords_counts).most_common()

    # print("Performing exact keyword spotting evaluation")
    p_at_10, p_at_n, eer = eval_keyword_spotting(
        sigmoid_output_dict_subset, word_to_id, exact_keywords_counts, exact_keywords_dict, args.analyze
        )

    print
    print("-"*79)
    print("Exact keyword spotting:")
    print("Average P@10: {:.4f}".format(p_at_10))
    print("Average P@N: {:.4f}".format(p_at_n))
    print("Average EER: {:.4f}".format(eer))
    # print("-"*79)

    # print
    # print("Performing semantic keyword spotting evaluation")
    p_at_10, p_at_n, eer = eval_keyword_spotting(
        sigmoid_output_dict_subset, word_to_id, semkeywords_counts, semkeywords_dict, args.analyze
        )

    # print
    print("-"*79)
    print("Semantic keyword spotting:")
    print("Average P@10: {:.4f}".format(p_at_10))
    print("Average P@N: {:.4f}".format(p_at_n))
    print("Average EER: {:.4f}".format(eer))
    print("-"*79)

    keyword_precision_counts = eval_semkeyword_exact_sem_counts(
        sigmoid_output_dict_subset, word_to_id, keywords, exact_keywords_dict,
        semkeywords_dict
        )
    p_at_n_accum = 0
    n_total_accum = 0
    n_exact_accum = 0
    n_sem_accum = 0
    # print
    for keyword in sorted(keyword_precision_counts):
        (n_total, n_exact, n_sem) = keyword_precision_counts[keyword] 
        # print("{}: {} exact and {} semantic out of {}".format(keyword, n_exact, n_sem, n_total))
        p_at_n_accum += float(n_exact + n_sem) / n_total
        n_total_accum += n_total
        n_exact_accum += n_exact
        n_sem_accum += n_sem
    print "Breakdown of exact and semantic matches"
    print("Average P@N:           {:.4f}".format(p_at_n_accum / len(keyword_precision_counts)))
    print("Average* P@N overall:  {:.4f}".format(float(n_exact_accum + n_sem_accum) / n_total_accum))
    print("Average* P@N exact:    {:.4f}".format(float(n_exact_accum ) / n_total_accum))
    print("Average* P@N semantic: {:.4f}".format(float(n_sem_accum) / n_total_accum))
    print("-"*79)



if __name__ == "__main__":
    main()
