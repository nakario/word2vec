#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
"""
import argparse
import collections
from typing import Dict
from typing import Iterable
from typing import NamedTuple
from typing import Union

import numpy as np

import chainer
from chainer import cuda
from chainer import Variable
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions
from chainer.utils import WalkerAlias

try:
    import cupy as cp
except ImportError:
    cp = np


ndarray = Union[np.ndarray, cp.ndarray]


class ConstArguments(NamedTuple):
    gpu: int
    unit: int
    ambiguity: int
    window: int
    batchsize: int
    epoch: int
    model: str
    negative_size: int
    out_type: str
    out: str
    test: bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=100, type=int,
                        help='number of units')
    parser.add_argument('--ambiguity', '-a', default=10, type=int,
                        help='ambiguity of words')
    parser.add_argument('--window', '-w', default=5, type=int,
                        help='window size')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                        default='skipgram',
                        help='model type ("skipgram", "cbow")')
    parser.add_argument('--negative-size', default=5, type=int,
                        help='number of negative samples')
    parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'],
                        default='hsm',
                        help='output model type ("hsm": hierarchical softmax, '
                        '"ns": negative sampling, '
                        '"original": no approximation)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)

    return ConstArguments(**vars(parser.parse_args()))


class CBoW_NS_VV(chainer.Chain):

    def __init__(
            self,
            n_vocab: int,
            n_units: int,
            n_sample: int,
            counts,
            power: float = 0.75,
            ignore_label: int = -1
    ):
        super(CBoW_NS_VV, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.embed_out = L.EmbedID(
                n_vocab, n_units, initialW=0
            )
            p = self.xp.array(counts, 'f')
            power = self.xp.float32(power)
            self.xp.power(p, power, p)  # p = self.xp.power(p, power)
            self.sampler = WalkerAlias(p)
            self.n_units = n_units
            self.n_sample = n_sample
            self.ignore_label = ignore_label

    def to_cpu(self):
        super(CBoW_NS_VV, self).to_cpu()
        self.sampler.to_cpu()

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(CBoW_NS_VV, self).to_gpu()
            self.sampler.to_gpu()

    def __call__(self, x: ndarray, context: ndarray):
        # x.shape == (batchsize,)
        # context.shape == (batchsize, context_size)
        shape = context.shape
        e = self.embed(context)
        # e.shape == (batchsize, context_size, n_units)
        h: Variable = F.sum(e, axis=1) * (1. / shape[1])
        # h.shape == (batchsize, n_units)
        samples = self.sampler.sample((shape[0], self.n_sample + 1))
        # samples.shape == (batchsize, n_sample + 1)
        samples[:, 0] = x
        w = self.embed_out(samples)
        # w.shape == (batchsize, n_sample + 1, n_units)
        wh = F.squeeze(F.matmul(w, h[:, :, None]))
        # wh.shape == (batchsize, n_sample + 1)
        signs = self.xp.ones(self.n_sample + 1, 'f')
        signs[0] = self.xp.float32(-1.)
        wh = F.scale(wh, signs, axis=1)
        y = F.sum(-F.log(F.sigmoid(wh)), axis=1)
        # y.shape == (batchsize,)
        ignore_mask = (x != self.ignore_label)
        loss = F.sum(F.where(ignore_mask, y, self.xp.zeros(shape[0], 'f')))
        # loss.shape == (batchsize,)
        reporter.report({'loss': loss}, self)
        return loss


class CBoW_NS_VM(chainer.Chain):

    def __init__(
            self,
            n_vocab: int,
            n_units: int,
            ambiguity: int,
            n_sample: int,
            counts,
            power: float = 0.75,
            ignore_label: int = -1
    ):
        super(CBoW_NS_VM, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.embed_out = L.EmbedID(
                n_vocab, n_units * ambiguity, initialW=0
            )
            p = self.xp.array(counts, 'f')
            power = self.xp.float32(power)
            self.xp.power(p, power, p)  # p = self.xp.power(p, power)
            self.sampler = WalkerAlias(p)
            self.n_units = n_units
            self.ambiguity = ambiguity
            self.n_sample = n_sample
            self.ignore_label = ignore_label

    def to_cpu(self):
        super(CBoW_NS_VM, self).to_cpu()
        self.sampler.to_cpu()

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(CBoW_NS_VM, self).to_gpu()
            self.sampler.to_gpu()

    def __call__(self, x: ndarray, context: ndarray):
        # x.shape == (batchsize,)
        # context.shape == (batchsize, context_size)
        shape = context.shape
        e = self.embed(context)
        # e.shape == (batchsize, context_size, n_units)
        h: Variable = F.sum(e, axis=1) * (1. / shape[1])
        h = F.broadcast_to(
            h[:, None, :, None],
            (shape[0], self.ambiguity, self.n_units, 1)
        )
        # h.shape == (batchsize, ambiguity, n_units, 1)
        samples = self.sampler.sample((shape[0], self.n_sample + 1))
        # samples.shape == (batchsize, n_sample + 1)
        samples[:, 0] = x
        w = self.embed_out(samples)
        # w.shape == (batchsize, n_sample + 1, n_units * ambiguity)
        w = F.reshape(
            w,
            (shape[0], self.n_sample + 1, self.ambiguity, self.n_units)
        )
        w = F.swapaxes(w, axis1=1, axis2=2)
        # w.shape == (batchsize, ambiguity, n_sample + 1, n_units)
        wh = F.squeeze(F.matmul(w, h))
        # wh.shape == (batchsize, ambiguity, n_sample + 1)
        signs = self.xp.ones(self.n_sample + 1, 'f')
        signs[0] = self.xp.float32(-1.)
        wh = F.scale(wh, signs, axis=2)
        eps = self.xp.float32(2e-5)
        y = F.sum(F.min(-F.log(F.sigmoid(wh) + eps), axis=1), axis=1)
        # y.shape == (batchsize,)
        ignore_mask = (x != self.ignore_label)
        loss = F.sum(F.where(ignore_mask, y, self.xp.zeros(shape[0], 'f')))
        # loss.shape == (batchsize,)
        reporter.report({'loss': loss}, self)
        return loss


class ContinuousBoW(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.loss_func = loss_func

    def __call__(self, x: ndarray, context: ndarray):
        # x.shape == (batchsize,)
        # context.shape == (batchsize, window * 2)
        e = self.embed(context)
        # e.shape == (batchsize, window * 2, n_units)
        h: Variable = F.sum(e, axis=1) * (1. / context.shape[1])
        # h.shape == (batchsize, n_units)
        loss = self.loss_func(h, x)
        reporter.report({'loss': loss}, self)
        return loss


class SkipGram(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.loss_func = loss_func

    def __call__(self, x: ndarray, context: ndarray):
        # x.shape == (batchsize,)
        # context.shape == (batchsize, window * 2)
        e = self.embed(context)
        # e.shape == (batchsize, window * 2, n_units)
        shape = e.shape
        x = F.broadcast_to(x[:, None], (shape[0], shape[1]))
        # x.shape == (batchsize, window * 2)
        e = F.reshape(e, (shape[0] * shape[1], shape[2]))
        # e.shape == (batchsize * window * 2, n_units)
        x = F.reshape(x, (shape[0] * shape[1],))
        # x.shape == (batchsize * window * 2,)
        loss = self.loss_func(e, x)
        reporter.report({'loss': loss}, self)
        return loss


class SoftmaxCrossEntropyLoss(chainer.Chain):

    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        with self.init_scope():
            self.out = L.Linear(n_in, n_out, initialW=0)

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


class WindowIterator(chainer.dataset.Iterator):

    def __init__(
            self,
            dataset: Union[np.ndarray, Iterable],
            window: int,
            batch_size: int,
            repeat: bool = True
    ):
        self.dataset = np.array(dataset, np.int32)
        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat

        self.order = np.random.permutation(
            len(dataset) - window * 2).astype(np.int32)
        self.order += window
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i: i_end]
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        pos = position[:, None] + offset[None, :]
        context = self.dataset.take(pos)
        center = self.dataset.take(position)

        if i_end >= len(self.order):
            np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return center, context

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)


def convert(batch, device):
    center, context = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        context = cuda.to_gpu(context)
    return center, context


def main():
    args = get_args()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Window: {}'.format(args.window))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Training model: {}'.format(args.model))
    print('Output type: {}'.format(args.out_type))
    print('')

    train, val, _ = chainer.datasets.get_ptb_words()
    train: np.ndarray = train
    val: np.ndarray = val
    counts = collections.Counter(train)
    counts.update(collections.Counter(val))
    n_vocab: int = max(train) + 1

    assert len(train.shape) == 1
    assert len(val.shape) == 1

    if args.test:
        train: np.ndarray = train[:100]
        val: np.ndarray = val[:100]

    vocab: Dict[str, int] = chainer.datasets.get_ptb_words_vocabulary()
    index2word: Dict[int, str] = {wid: word for word, wid in vocab.items()}

    print('n_vocab: %d' % n_vocab)
    print('data length: %d' % len(train))

    if args.out_type == 'hsm':
        HSM = L.BinaryHierarchicalSoftmax
        tree = HSM.create_huffman_tree(counts)
        loss_func = HSM(args.unit, tree)
        loss_func.W.data[...] = 0
    elif args.out_type == 'ns':
        cs = [counts[w] for w in range(len(counts))]
        loss_func = L.NegativeSampling(args.unit, cs, args.negative_size)
        loss_func.W.data[...] = 0
    elif args.out_type == 'original':
        loss_func = SoftmaxCrossEntropyLoss(args.unit, n_vocab)
    else:
        raise Exception('Unknown output type: {}'.format(args.out_type))

    if args.model == 'skipgram':
        model = SkipGram(n_vocab, args.unit, loss_func)
    elif args.model == 'cbow':
        cs = [counts[w] for w in range(len(counts))]
        model = CBoW_NS_VM(n_vocab, args.unit, args.ambiguity,
                           args.negative_size, cs)
    else:
        raise Exception('Unknown model type: {}'.format(args.model))

    if args.gpu >= 0:
        model.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(model)

    train_iter = WindowIterator(train, args.window, args.batchsize)
    val_iter = WindowIterator(val, args.window, args.batchsize, repeat=False)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(
        val_iter, model, converter=convert, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    with open('word2vec.model', 'w') as f:
        f.write('%d %d\n' % (len(index2word), args.unit))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))


if __name__ == '__main__':
    main()
