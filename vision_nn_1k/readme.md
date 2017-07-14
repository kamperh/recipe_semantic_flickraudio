Vision Neural Networks Trained on a Common 1k Vocabulary
========================================================

Get and apply VGG-16 to Flickr8k, Flickr30k and MSCOCO images
-------------------------------------------------------------
Download the VGG-16 weights in Numpy archive format:

    cd data/
    wget https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
    cd ..

When applying VGG-16, we found that cropping instead of naive rescaling to
(244, 244) improves performance slightly, so in all cases we pass the `--crop`
argument to `apply_vgg16.py`.

Apply VGG-16 to Flickr8k:

    mkdir -p data/flickr8k/
    ./apply_vgg16.py --crop --output_layer fc7 --batch_size 31 \
        /share/data/lang/users/kamperh/flickr_multimod/Flickr8k_Dataset/Flicker8k_Dataset/ \
        data/flickr8k/fc7.npz


Apply VGG-16 to Flickr30k:

    mkdir -p data/flickr30k/
    ./apply_vgg16.py --crop --output_layer fc7 --batch_size 37 \
        /share/data/vision-greg/flickr30k/flickr30k-images/ \
        data/flickr30k/fc7.npz


Apply VGG-16 to the MSCOCO train, val and test splits:

    mkdir -p data/mscoco/train
    ./apply_vgg16.py --crop --batch_size 19 \
        /share/data/lang/users/spsettle/tf-common/multimodal/datasets/mscoco/coco/train2014 \
        data/mscoco/train/fc7.npz

    mkdir -p data/mscoco/val
    ./apply_vgg16.py --crop --batch_size 61 \
        /share/data/lang/users/spsettle/tf-common/multimodal/datasets/mscoco/coco/val2014 \
        data/mscoco/val/fc7.npz

    mkdir -p data/mscoco/test
    ./apply_vgg16.py --crop --batch_size 35 \
        /share/data/lang/users/spsettle/tf-common/multimodal/datasets/mscoco/coco/test2014 \
        data/mscoco/test/fc7.npz


Prepare datasets in common format
---------------------------------
Prepare datasets without processing captions:

    ./data_prep_flickr8k.py
    ./data_prep_flickr30k.py
    ./data_prep_mscoco.py
    ./data_prep_mscoco+flickr30k.py

Process the captions for MSCOCO+Flickr30k, defining the common vocabulary for
all systems and testing:

    ./get_captions_word_ids.py data/mscoco+flickr30k/captions.txt \
        data/mscoco+flickr30k/

Using this common word to ID mapping, process the captions for the other
datasets:

    ./get_captions_using_word_ids.py \
        data/flickr8k/captions.txt \
        data/mscoco+flickr30k/word_to_id_content.pkl \
        data/flickr8k/captions_word_ids_content_dict.pkl
    ./get_captions_using_word_ids.py \
        data/flickr30k/captions.txt \
        data/mscoco+flickr30k/word_to_id_content.pkl \
        data/flickr30k/captions_word_ids_content_dict.pkl
    ./get_captions_using_word_ids.py \
        data/mscoco/captions.txt \
        data/mscoco+flickr30k/word_to_id_content.pkl \
        data/mscoco/captions_word_ids_content_dict.pkl


Bag-of-words MLP on top of VGG-16
---------------------------------
To change the dataset, change the parameters in `default_options_dict` in
`train_bow_mlp.py`. Train a bag-of-words MLP on top of VGG-16 features:

    ./train_bow_mlp.py

Apply and evaluate this model on the Flickr30k development set:

    ./apply_bow_mlp.py models/flickr30k/train_bow_mlp/153210e6ef \
        flickr30k dev
    ./eval_precision_recall.py --analyze --sigmoid_threshold 0.4 \
        models/flickr30k/train_bow_mlp/153210e6ef flickr30k dev

Apply and evaluate this model on the MSCOCO development set:

    ./apply_bow_mlp.py models/flickr30k/train_bow_mlp/153210e6ef \
        mscoco val
    ./eval_precision_recall.py --analyze --sigmoid_threshold 0.4 \
        models/flickr30k/train_bow_mlp/153210e6ef mscoco val

To apply the model to Flickr8k, and inspect predictions, run:

    ./apply_bow_mlp.py models/flickr30k/train_bow_mlp/153210e6ef \
        flickr8k all
    ./show_predictions.py --sigmoid_threshold 0.7 \
        models/flickr30k/train_bow_mlp/153210e6ef/sigmoid_output_dict.flickr8k.all.npz


References
----------
VGG-16 code is based on <https://www.cs.toronto.edu/~frossard/post/vgg16/>.
Flickr30k is split into train/dev/test following the same split used in
<http://cs.stanford.edu/people/karpathy/deepimagesent/>. The splits are given
in the `../data/` directory.

