#!/usr/bin/env python

"""
Prepare the Flickr30k data splits and labels.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from __future__ import print_function
from os import path
from shutil import copyfile
import re
import sys

sys.path.append("..")

from paths import flickr30k_dir

output_dir = path.join("data", "flickr30k")


def main():

    flickr30k_captions_fn = path.join(flickr30k_dir, "results_20130124.token")
    captions_dict = {}
    print("Reading:", flickr30k_captions_fn)
    with open(flickr30k_captions_fn) as f:
        for line in f:
            line = line.strip().split()
            utt = line[0].replace(".jpg#", "_")
            captions_dict[utt] = " ".join(
                [i.lower() for i in line[1:] if re.match(r"^[a-z\-']*$", i.lower())]
                )
    captions_fn = path.join(output_dir, "captions.txt")
    print("Writing:", captions_fn)
    with open(captions_fn, "w") as f:
        for utt_label in sorted(captions_dict):
            f.write(utt_label + " " + captions_dict[utt_label] + "\n")
    # # Copy captions to captions.txt
    # src_fn = path.join(flickr30k_dir, "results_20130124.token")
    # dest_fn = path.join(output_dir, "captions.txt")
    # print("Copying", src_fn, "to", dest_fn)
    # copyfile(src_fn, dest_fn)

    # Copy data splits
    src_fn = path.join("..", "data", "flickr30k_all_no8k.txt")
    dest_fn = path.join(output_dir, "all_no8k.txt")
    print("Copying", src_fn, "to", dest_fn)
    copyfile(src_fn, dest_fn)
    src_fn = path.join("..", "data", "flickr30k_train.txt")
    dest_fn = path.join(output_dir, "train.txt")
    print("Copying", src_fn, "to", dest_fn)
    copyfile(src_fn, dest_fn)
    src_fn = path.join("..", "data", "flickr30k_dev.txt")
    dest_fn = path.join(output_dir, "dev.txt")
    print("Copying", src_fn, "to", dest_fn)
    copyfile(src_fn, dest_fn)
    src_fn = path.join("..", "data", "flickr30k_test.txt")
    dest_fn = path.join(output_dir, "test.txt")
    print("Copying", src_fn, "to", dest_fn)
    copyfile(src_fn, dest_fn)


if __name__ == "__main__":
    main()
