# Fork of chainer/chainer/examples/word2vec

This is a fork of chainer's example.
See [LICENSES/chainer](https://github.com/nakario/word2vec/blob/master/LICENSES/chainer) to check the original license.


The following is the original README
* * *

# Word Embedding

This is an example of word embedding.
We implemented Mikolov's Skip-gram model and Continuous-BoW model with Hierarchical softmax and Negative sampling.

Run `train_word2vec.py` to train and get `word2vec.model` which includes embedding data.
You can find top-5 nearest embedding vectors using `search.py`.

This example is based on the following word embedding implementation in C++.
https://code.google.com/p/word2vec/
