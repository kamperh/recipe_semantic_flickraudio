Semantic Speech Retrieval using the Flickr Audio Captions Corpus
================================================================

Overview
--------
This is a recipe for training a model on images paired with untranscribed
speech, and using this model for semantic keyword spotting. The model and
this new task are described in the following publications:

- H. Kamper, G. Shakhnarovich, and K. Livescu, "Semantic speech retrieval with
  a  visually grounded model of untranscribed speech,"  IEEE/ACM Transactions
  on Audio, Speech and Language Processing, vol. 27, no. 1, pp. 89-98, 2019.
  [[arXiv](https://arxiv.org/abs/1710.01949)]
- H. Kamper, S. Settle, G. Shakhnarovich, and K. Livescu, "Visually grounded
  learning of keyword prediction from untranscribed speech," in *Proc.
  Interspeech*, 2017. [[arXiv](https://arxiv.org/abs/1706.03818)]

Please cite these papers if you use the code.


Related repositories and datasets
---------------------------------
A related [recipe](https://github.com/kamperh/recipe_vision_speech_flickr) is
also available, but this one is most recent recipe.

The semantic labels used here are also available separately in the
[semantic_flickraudio](https://github.com/kamperh/semantic_flickraudio)
repository. Here we directly use processed versions of this dataset: all the
pickled files in `data/` starting with `06-16-23h59` were obtained directly
from the semantic annotations.

The output of the multilabel visual classifier described below (also see
[vision_nn_1k/readme.md](vision_nn_1k/readme.md)) can be downloaded directly
[here](https://github.com/JSALT-Rosetta/flickr/blob/master/flickr8k.tags.all.txt.zip).
We released these visual tags as part of the
[JSALT Rosetta](https://github.com/JSALT-Rosetta/flickr) project.


Disclaimer
----------
The code provided here is not pretty. But I believe research should be
reproducible, and I hope that this repository is sufficient to make this
possible for the above paper. I provide no guarantees with the code, but please
let me know if you have any problems, find bugs or have general comments.


Datasets
--------
The following datasets need to be obtained:

- [Flickr audio](https://groups.csail.mit.edu/sls/downloads/flickraudio/)
- [Flickr8k images](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip)
- [Flickr8k text](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip)
- [Flickr30k](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/)
- [MSCOCO](http://cocodataset.org/#download)

MSCOCO and Flickr30k is used for training a vision tagging system. The Flickr8k
audio and image datasets gives paired images with spoken captions; we do not
use the labels from either of these. The Flickr8k text corpus is purely for
reference. The Flickr8k dataset can also be browsed directly
[here](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html).


Directory structure
-------------------
- `data/` - Contains permanent data (file lists, annotations) that are used
  elsewhere.
- `speech_nn/` - Speech systems trained on the Flickr Audio Captions Corpus.
- `vision_nn_1k/` - Vision systems trained on Flickr30k, MSCOCO and
  Flickr30k+MSCOCO, but with the vocabulary given by the 1k most common words
  in Flickr30k+MSCOCO. Evaluation is also only for those 1k words.


Preliminary
-----------
Install all the standalone dependencies (below). Then clone the required GitHub
repositories into `../src/` as follows:

    mkdir ../src/
    git clone https://github.com/kamperh/tflego.git ../src/tflego/

Download all the required datasets (above), and then update `paths.py` to point
to the corresponding directories.


Feature extraction
------------------
Extract filterbank and MFCC features by running the steps in
[kaldi_features/readme.md](kaldi_features/readme.md).


Neural network training
-----------------------
Train the multi-label visual classifier by running the steps in
[vision_nn_1k/readme.md](vision_nn_1k/readme.md). Note the final model
directory.

Train the various visually grounded speech models by running the steps in
[speech_nn/readme.md](speech_nn/readme.md).


Dependencies
------------
Standalone packages:

- [Python](https://www.python.org/): I used Python 2.7.
- [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/).
- [TensorFlow](https://www.tensorflow.org/): Required by the `tflego`
  repository below. I used TensorFlow v0.10.
- [Kaldi](http://kaldi-asr.org/): Used for feature extraction.

Repositories from GitHub:

- [tflego](https://github.com/kamperh/tflego): A wrapper for building neural
  networks. Should be cloned into the directory `../src/tflego/`.


Contributors
------------
- [Herman Kamper](http://www.kamperh.com/)
- [Shane Settle](https://github.com/shane-settle)
- [Karen Livescu](http://ttic.uchicago.edu/~klivescu/)
- [Greg Shakhnarovich](http://ttic.uchicago.edu/~gregory/)


License
-------
The code is distributed under the Creative Commons Attribution-ShareAlike
license ([CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)).
