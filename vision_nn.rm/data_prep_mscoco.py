#!/usr/bin/env python

"""
Prepare the MSCOCO dataset.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from datetime import datetime
from os import path
import json
import numpy as np
import re
import sys

sys.path.append("..")

from paths import mscoco_annotations_dir

output_dir = path.join("data", "mscoco")


def process_caption(caption):
    caption = caption.replace("\n", "")
    caption = re.sub('[0-9\!\,\.\"\;\?\:\#\@\(\)\$\+\=\[\]\_\>\%\*\\\`"]', "", caption)
    caption = caption.replace("/", " ").replace("&", "and").replace("_", " ").lower()
    return caption


def get_captions_urls(json):

    # Get the captions for each image
    image_id_to_captions = {}
    for i_entry in xrange(len(json["annotations"])):
        image_id = json["annotations"][i_entry]["image_id"]
        caption = json["annotations"][i_entry]["caption"]
        if not image_id in image_id_to_captions:
            image_id_to_captions[image_id] = []
        image_id_to_captions[image_id].append(process_caption(caption))

    # Get the URL for each image
    captions_dict = {}
    url_dict = {}
    for i_entry in xrange(len(json["images"])):
        image_id = json["images"][i_entry]["id"]
        mscoco_fn = json["images"][i_entry]["file_name"]
        mscoco_label = path.splitext(mscoco_fn)[0]
        url = json["images"][i_entry]["url"]
        url_dict[mscoco_label] = url
        for i_caption in xrange(len(image_id_to_captions[image_id])):
            captions_dict[mscoco_label + "_" + str(i_caption)] = image_id_to_captions[image_id][i_caption]

    return captions_dict, url_dict


def main():

    print(datetime.now())

    if not path.isdir(path.join("data", "mscoco")):
        os.makedirs(path.join("data", "mscoco"))

    # Combine feature files
    combined_npz_fn = path.join(output_dir, "fc7.npz")
    if not path.isfile(combined_npz_fn):
        combined_dict = {}
        for subset in ["train", "val"]:
            npz_fn = path.join(output_dir, subset, "fc7.npz")
            print("Reading: " + npz_fn)
            features = np.load(npz_fn)
            for image_label in features:
                combined_dict[image_label] = features[image_label]
        print("Writing: " + combined_npz_fn)
        np.savez(combined_npz_fn, **combined_dict)
        print("No. images: " + str(len(combined_dict)))

    print(datetime.now())

    # Obtain captions and URLs
    captions_json_fn = path.join(mscoco_annotations_dir, "captions_val2014.json")
    print("Reading: " + captions_json_fn)
    with open(captions_json_fn) as f:
        captions_json = json.load(f)
    val_captions, val_urls = get_captions_urls(captions_json)
    captions_json_fn = path.join(mscoco_annotations_dir, "captions_train2014.json")
    print("Reading: " + captions_json_fn)
    with open(captions_json_fn) as f:
        captions_json = json.load(f)
    train_captions, train_urls = get_captions_urls(captions_json)

    # Write lists of training and validation images
    fn = path.join(output_dir, "train.txt")
    print("Writing: " + fn)
    with open(fn, "w") as f:
        for image_label in train_urls:
            f.write(image_label + "\n")
    fn = path.join(output_dir, "val.txt")
    print("Writing: " + fn)
    with open(fn, "w") as f:
        for image_label in val_urls:
            f.write(image_label + "\n")

    # Write URLs
    fn = path.join(output_dir, "trainval_to_url.txt")
    print("Writing: " + fn)
    with open(fn, "w") as f:
        for image_label in sorted(train_urls):
            f.write(image_label + " " + train_urls[image_label] + "\n")
        for image_label in sorted(val_urls):
            f.write(image_label + " " + val_urls[image_label] + "\n")

    # Write captions
    fn = path.join(output_dir, "captions.txt")
    print("Writing: " + fn)
    with open(fn, "w") as f:
        for image_label in sorted(train_captions):
            f.write(image_label + " " + train_captions[image_label] + "\n")
        for image_label in sorted(val_captions):
            f.write(image_label + " " + val_captions[image_label] + "\n")

    print(datetime.now())

if __name__ == "__main__":
    main()
