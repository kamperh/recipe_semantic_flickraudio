#!/usr/bin/env python

"""
Prepare the Flickr8k dataset.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from __future__ import print_function
from os import path
import os
import sys

sys.path.append("..")

from paths import flickr8k_text_dir

captions_fn = path.join("..", "kaldi_features", "data", "full_vad", "text")
output_dir = path.join("data", "flickr8k")


def get_flickr8k_train_test_dev():
    set_dict = {}
    for subset in ["train", "dev", "test"]:
        if subset not in set_dict:
            set_dict[subset] = []
        subset_fn = path.join(flickr8k_text_dir, "Flickr_8k.{}Images.txt".format(subset))
        print("Reading:", subset_fn)
        with open(subset_fn) as f:
            for line in f:
                set_dict[subset].append(path.splitext(line.strip())[0])
    return set_dict


def main():

    if not path.isdir(output_dir):
        os.makedirs(output_dir)

    print("Reading:", captions_fn)
    captions_dict = {}
    with open(captions_fn) as f:
        for line in f:
            line = line.strip().split()
            captions_dict[line[0]] = [i for i in line[1:] if "<" not in i and not "'" in i]
    fn = path.join(output_dir, "captions.txt")
    print("Writing:", fn)
    with open(fn, "w") as f:
        for utt_label in sorted(captions_dict):
            f.write(utt_label[4:] + " " + " ".join(captions_dict[utt_label]) + "\n")

    flickr8k_sets = get_flickr8k_train_test_dev()
    for subset in ["train", "dev", "test"]:
        fn = path.join(output_dir, subset + ".txt")
        print("Writing:", fn)
        with open(fn, "w") as f:
            for image_label in sorted(flickr8k_sets[subset]):
                f.write(image_label + "\n")
    fn = path.join(output_dir, "all.txt")    
    print("Writing:", fn)
    with open(fn, "w") as f:
        for subset in ["train", "dev", "test"]:
            for image_label in sorted(flickr8k_sets[subset]):
                f.write(image_label + "\n")


if __name__ == "__main__":
    main()
