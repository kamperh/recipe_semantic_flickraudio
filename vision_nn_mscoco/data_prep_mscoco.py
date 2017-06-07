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
import re
import sys

sys.path.append("..")

from paths import mscoco_annotations_dir, flickr8k_text_dir


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
    fn = path.join("data", "mscoco", "train.txt")
    print("Writing: " + fn)
    with open(fn, "w") as f:
        for image_label in train_urls:
            f.write(image_label + "\n")
    fn = path.join("data", "mscoco", "val.txt")
    print("Writing: " + fn)
    with open(fn, "w") as f:
        for image_label in val_urls:
            f.write(image_label + "\n")

    # Write URLs
    fn = path.join("data", "mscoco", "trainval_to_url.txt")
    print("Writing: " + fn)
    with open(fn, "w") as f:
        for image_label in sorted(train_urls):
            f.write(image_label + " " + train_urls[image_label] + "\n")
        for image_label in sorted(val_urls):
            f.write(image_label + " " + val_urls[image_label] + "\n")

    # Write captions
    fn = path.join("data", "mscoco", "captions_trainval.txt")
    print("Writing: " + fn)
    with open(fn, "w") as f:
        for image_label in sorted(train_captions):
            f.write(image_label + " " + train_captions[image_label] + "\n")
        for image_label in sorted(val_captions):
            f.write(image_label + " " + val_captions[image_label] + "\n")

    # # Write captions
    # print("Reading: " + mscoco_json_fn)
    # with open(mscoco_json_fn) as f:
    #     mscoco_json = json.load(f)
    # fn = path.join("data", "mscoco", "captions_trainval.txt")
    # print("Writing: " + fn)
    # with open(fn, "w") as f:
    #     for i in xrange(len(mscoco_json)):
    #         image_label = path.splitext(path.split(mscoco_json[i]["file_path"])[-1])[0]
    #         captions = mscoco_json[i]["captions"]
    #         for i_caption in xrange(len(captions)):
    #             caption = captions[i_caption].replace("\n", "")
    #             caption = re.sub('[0-9\!\,\.\"\;\?\:\#\@\(\)\$\+\=\[\]\_\>\%\*\\\`"]', "", caption)
    #             caption = caption.replace("/", " ").replace("&", "and").replace("_", " ").lower()
    #             f.write(image_label + "_" + str(i_caption) + " " + caption + "\n")

    # # Write mapping to original Flickr url
    # mscoco_to_url = []
    # for subset in ["train", "val"]:
    #     captions_json_fn = path.join(mscoco_annotations_dir, "captions_" + subset + "2014.json")
    #     print("Reading: " + captions_json_fn)
    #     with open(captions_json_fn) as f:
    #         captions_json = json.load(f)
    #     for i in xrange(len(captions_json["images"])):
    #         image_label = path.splitext(captions_json["images"][i]["file_name"])[0]
    #         url = captions_json["images"][i]["url"]
    #         mscoco_to_url.append((image_label, url))
    # fn = path.join("data", "mscoco", "trainval_to_url.txt")
    # print("Writing: " + fn)
    # with open(fn, "w") as f:
    #     for image_label, url in mscoco_to_url:
    #         f.write(image_label + " " + url + "\n")

    # # Remove overlapping images from Flickr8k
    # flickr8k_images = []
    # for subset in ["train", "dev", "test"]:
    #     fn = path.join(flickr8k_text_dir, "Flickr_8k.{}Images.txt".format(subset))
    #     print("Reading: " + fn)
    #     with open(fn) as f:
    #         for line in f:
    #             line = path.splitext(line.strip())[0]
    #             flickr8k_images.append(line)
    # trainval_no_flickr8k = []
    # for image_label, url in mscoco_to_url:
    #     url_base = path.splitext(path.split(url)[-1])[0][:-2]
    #     if url_base in flickr8k_images:
    #         print("MSCOCO image in Flickr8k: " + url_base)
    #     else:
    #         trainval_no_flickr8k.append(image_label)
    # fn = path.join("data", "mscoco", "trainval_no_flickr8k.txt")
    # print("Writing: " + fn)
    # with open(fn, "w") as f:
    #     for image_label in trainval_no_flickr8k:
    #         f.write(image_label + "\n")
    # fn_train = path.join("data", "mscoco", "train_no_flickr8k.txt")
    # fn_val = path.join("data", "mscoco", "val_no_flickr8k.txt")
    # print("Writing: " + fn_train)
    # print("Writing: " + fn_val)
    # with open(fn_train, "w") as f_train, open(fn_val, "w") as f_val:
    #     for image_label in trainval_no_flickr8k:
    #         if "train" in image_label:
    #             f_train.write(image_label + "\n")
    #         elif "val" in image_label:
    #             f_val.write(image_label + "\n")

    # print(datetime.now())


if __name__ == "__main__":
    main()
