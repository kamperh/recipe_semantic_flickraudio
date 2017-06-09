#!/usr/bin/env python

"""
Prepare the combined MSCOCO+Flickr30k dataset.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from datetime import datetime
from os import path
from shutil import copyfile
import numpy as np
import os
import re
import sys

sys.path.append("..")

output_dir = path.join("data", "mscoco+flickr30k")
mscoco_npz_fn = path.join("data", "mscoco", "fc7.npz")
mscoco_train_list_fn = path.join("data", "mscoco", "train.txt")
mscoco_val_list_fn = path.join("data", "mscoco", "val.txt")
mscoco_captions_fn = path.join("data", "mscoco", "captions.txt")
flickr30k_npz_fn = path.join("data", "flickr30k", "fc7.npz")
flickr30k_train_list_fn = path.join("data", "flickr30k", "all_no8k.txt")
flickr30k_captions_fn = path.join("data", "flickr30k", "captions.txt")


def main():

    print(datetime.now())

    # Create output directory
    if not path.isdir(output_dir):
        os.makedirs(output_dir)

    # Combine captions
    captions_fn = path.join(output_dir, "captions.txt")
    if not path.isfile(captions_fn):
        print("Reading: " + mscoco_captions_fn)
        captions_dict = {}
        with open(mscoco_captions_fn) as f:
            for line in f:
                line = line.strip().split()
                utt_label = line[0]
                caption = " ".join(line[1:])
                captions_dict[utt_label] = caption
        print("Reading: " + flickr30k_captions_fn)
        with open(flickr30k_captions_fn) as f:
            for line in f:
                line = line.strip().split()
                utt = line[0].replace(".jpg#", "_")
                captions_dict[utt] = " ".join(
                    [i.lower() for i in line[1:] if re.match(r"^[a-z\-']*$", i.lower())]
                    )
        print("Writing: "+ captions_fn)
        with open(captions_fn, "w") as f:
            for utt_label in sorted(captions_dict):
                f.write(utt_label + " " + captions_dict[utt_label] + "\n")
    else:
        print("Existing captions: " + captions_fn)

    print(datetime.now())

    # Combine Numpy archives
    combined_npz_fn = path.join(output_dir, "fc7.npz")
    if not path.isfile(combined_npz_fn):
        print("Reading: " + mscoco_npz_fn)
        mscoco_npz = np.load(mscoco_npz_fn)
        print("Reading: " + flickr30k_npz_fn)
        flickr30k_npz = np.load(flickr30k_npz_fn)
        combined_dict = {}
        for image_label in mscoco_npz:
            combined_dict[image_label] = mscoco_npz[image_label]
        for image_label in flickr30k_npz:
            combined_dict[image_label] = flickr30k_npz[image_label]
        print("Writing: " + combined_npz_fn)
        np.savez(combined_npz_fn, **combined_dict)
        print("No. images: " + str(len(combined_dict)))
    else:
        print("Existing Numpy archive: " + combined_npz_fn)


    print(datetime.now())

    # Write training and validation lists
    train_list_fn = path.join(output_dir, "train.txt")
    if not path.isfile(train_list_fn):
        ignore_images = []
        fn = path.join("..", "data", "mscoco_in_flickr8k.txt")
        print("Reading: " + fn)
        with open(fn) as f:
            for line in f:
                ignore_images.append(line.strip().split()[0])
        train_list = []
        print("Reading: " + mscoco_train_list_fn)
        with open(mscoco_train_list_fn) as f:
            for line in f:
                image_label = line.strip()
                if image_label not in ignore_images:
                    train_list.append(image_label)
        print("Reading: " + flickr30k_train_list_fn)
        with open(flickr30k_train_list_fn) as f:
            for line in f:
                train_list.append(line.strip())     
        print("Writing: " + train_list_fn)
        with open(train_list_fn, "w") as f:
            for image_label in sorted(train_list):
                f.write(image_label + "\n")
    else:
        print("Existing list: " + train_list_fn)
    val_list_fn = path.join(output_dir, "val.txt")
    if not path.isfile(val_list_fn):
        print("Copying: {} to {}".format(mscoco_val_list_fn, val_list_fn))
        copyfile(mscoco_val_list_fn, val_list_fn)
    else:
        print("Existing list: " + val_list_fn)

    print(datetime.now())


if __name__ == "__main__":
    main()
