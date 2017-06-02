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

def eval_semkeyword_exact_sem_counts(sigmoid_dict, word_to_id, keyword_counts,
        exact_label_dict, sem_label_dict):
    """
    Return a dict with elements `[n_total, n_exact, n_sem]` for each keyword.
    """

    keywords = sorted(keyword_counts)
    utterances = sorted(sigmoid_dict)
    keyword_ids = [word_to_id[w] for w in keywords]

    # Get sigmoid matrix for keywords
    keyword_sigmoid_mat = np.zeros((len(utterances), len(keywords)))
    for i_utt, utt in enumerate(utterances):
        keyword_sigmoid_mat[i_utt, :] = sigmoid_dict[utt][keyword_ids]

    for i_keyword, keyword in enumerate(keywords):

        # Rank
        rank_order = keyword_sigmoid_mat[:, i_keyword].argsort()[::-1]
        utt_order = [utterances[i] for i in rank_order]


    return
    
    keywords = sorted(keyword_counts)
    utterances = sorted(sigmoid_dict)
    keyword_ids = [word_to_id[w] for w in keywords]

    # Get sigmoid matrix for keywords
    keyword_sigmoid_mat = np.zeros((len(utterances), len(keywords)))
    for i_utt, utt in enumerate(utterances):
        keyword_sigmoid_mat[i_utt, :] = sigmoid_dict[utt][keyword_ids]

    # Keyword spotting evaluation
    p_at_10 = []
    p_at_n = []
    eer = []
    if analyze:
        print
    for i_keyword, keyword in enumerate(keywords):

        # Rank
        rank_order = keyword_sigmoid_mat[:, i_keyword].argsort()[::-1]
        utt_order = [utterances[i] for i in rank_order]

        # EER
        y_true = []
        for utt in utt_order:
            if keyword in label_dict[utt]:
                y_true.append(1)
            else:
                y_true.append(0)
        y_score = keyword_sigmoid_mat[:, i_keyword][rank_order]
        cur_eer = calculate_eer(y_true, y_score)
        eer.append(cur_eer)

        # P@10
        cur_p_at_10 = float(sum(y_true[:10]))/10.
        p_at_10.append(cur_p_at_10)

        # P@N
        cur_p_at_n = float(sum(y_true[:keyword_counts[keyword]]))/keyword_counts[keyword]
        p_at_n.append(cur_p_at_n)

        if analyze:
            print "-"*79
            print "Keyword:", keyword
            print "Current P@10: {:.4f}".format(cur_p_at_10)
            print "Current P@N: {:.4f}".format(cur_p_at_n)
            print "Current EER: {:.4f}".format(cur_eer)
            print "Top 10 utterances: ", utt_order[:10]
            if cur_p_at_10 != 1:
                # print "Incorrect in top 10:", [
                #     utt for i, utt in enumerate(utt_order[:10]) if y_true[i] == 0
                #     ]
                print "Incorrect in top 10:"
                if utt.count("_") == 3:
                    print "\n".join([
                        "/share/data/lang/users/kamperh/flickr_multimod/flickr_audio/wavs/"
                        + utt[4:] + ".wav" for i, utt in enumerate(utt_order[:10]) if y_true[i] == 0
                        ])
                elif utt.count("_") == 2:
                    print "\n".join([
                        "/share/data/lang/users/kamperh/flickr_multimod/flickr_audio/wavs/"
                        + utt + ".wav" for i, utt in enumerate(utt_order[:10]) if y_true[i] == 0
                        ])
                else:
                    assert False

    if analyze:
        print "-"*79
        print

    # Average
    p_at_10 = np.mean(p_at_10)
    p_at_n = np.mean(p_at_n)
    eer = np.mean(eer)

    return p_at_10, p_at_n, eer


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
    print("-"*79)

    # print
    # print("Performing semantic keyword spotting evaluation")
    p_at_10, p_at_n, eer = eval_keyword_spotting(
        sigmoid_output_dict_subset, word_to_id, semkeywords_counts, semkeywords_dict, args.analyze
        )

    print
    print("-"*79)
    print("Semantic keyword spotting:")
    print("Average P@10: {:.4f}".format(p_at_10))
    print("Average P@N: {:.4f}".format(p_at_n))
    print("Average EER: {:.4f}".format(eer))
    print("-"*79)

    print eval_semkeyword_exact_sem_counts(sigmoid_output_dict_subset, word_to_id, semkeywords_counts,
        exact_keywords_dict, semkeywords_dict)


if __name__ == "__main__":
    main()
