#!/usr/bin/env python

"""
Convert captions to word IDs but use a given word IDs file.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from __future__ import print_function
from datetime import datetime
from os import path
import argparse
import cPickle as pickle
import re
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("captions_fn", type=str, help="captions file")
    parser.add_argument("word_to_id_fn", type=str, help="word to ID mapping file")
    parser.add_argument(
        "output_pkl", type=str, help="the converted caption dictionary is written to this file"
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
    
    captions_dict = {}
    print(datetime.now())
    print("Reading:", args.captions_fn)
    with open(args.captions_fn) as f:
        for line in f:
            line = line.strip().split()
            utt = line[0].replace(".jpg#", "_")
            captions_dict[utt] = [i.lower() for i in line[1:] if re.match(r"^[a-z\-']*$", i.lower())]
    print(datetime.now())

    print("Reading:", args.word_to_id_fn)
    with open(args.word_to_id_fn, "rb") as f:
        word_to_id = pickle.load(f)

    # Map captions
    captions_word_ids_dict = {}
    for utt in captions_dict:
        captions_word_ids_dict[utt] = [
            word_to_id[word] for word in captions_dict[utt] if word in word_to_id
            ]
    print("Writing:", args.output_pkl)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(captions_word_ids_dict, f, -1)


if __name__ == "__main__":
    main()
