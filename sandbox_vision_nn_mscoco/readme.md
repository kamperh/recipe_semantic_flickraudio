Vision Neural Networks Trained on MSCOCO
========================================


Preliminaries
-------------
The steps in [vision_nn_flickr30k/readme.md](../vision_nn_flickr30k/readme.md)
need to be executed first. Some scripts from that directory is also used here.

Prepare the MSCOCO dataset:

    ./data_prep_mscoco.py

Convert the MSCOCO captions to word IDs:

    ./get_captions_word_ids.py data/mscoco/captions_trainval.txt data/mscoco/

Prepare the combined MSCOCO+Flickr30k dataset and convert the captions to word
IDs:

    ./data_prep_mscoco+flickr30k.py
    ./get_captions_word_ids.py \
        data/mscoco+flickr30k/captions_trainval.txt \
        data/mscoco+flickr30k/


Apply pretrained VGG-16 to MSCOCO
---------------------------------
Apply VGG-16 to the MSCOCO training, validation and test images:

    mkdir -p data/mscoco/train
    mkdir -p data/mscoco/val
    mkdir -p data/mscoco/test
    
    ../vision_nn_flickr30k/apply_vgg16.py --crop --batch_size 19 \
        /share/data/lang/users/spsettle/tf-common/multimodal/datasets/mscoco/coco/train2014 \
        data/mscoco/train/fc7.npz

    ../vision_nn_flickr30k/apply_vgg16.py --crop --batch_size 61 \
        /share/data/lang/users/spsettle/tf-common/multimodal/datasets/mscoco/coco/val2014 \
        data/mscoco/val/fc7.npz

    ../vision_nn_flickr30k/apply_vgg16.py --crop --batch_size 35 \
        /share/data/lang/users/spsettle/tf-common/multimodal/datasets/mscoco/coco/test2014 \
        data/mscoco/test/fc7.npz

    ../vision_nn_flickr30k/apply_vgg16.py --crop --batch_size 31 \
        /share/data/lang/users/kamperh/flickr_multimod/Flickr8k_Dataset/Flicker8k_Dataset/ \
        data/flickr8k/fc7.npz

    ../vision_nn_flickr30k/apply_vgg16.py --output_layer fc7 --crop --batch_size 37 \
        /share/data/vision-greg/flickr30k/flickr30k-images/ \
        data/flickr30k/fc7.npz


Bag-of-words MLP for Flickr30k on top of VGG-16
-----------------------------------------------
Train a bag-of-words MLP on MSCOCO VGG-16 features:

    ./train_bow_mlp.py

Apply this to the Flickr8k dataset:

    ./apply_bow_mlp.py --batch_size 93 models/train_bow_mlp/9d3f0dca3b flickr8k
    ./show_predictions.py --sigmoid_threshold 0.7 \
        models/train_bow_mlp/9d3f0dca3b/sigmoid_output_dict.flickr8k.npz \
        data/mscoco/word_to_id_content.pkl

