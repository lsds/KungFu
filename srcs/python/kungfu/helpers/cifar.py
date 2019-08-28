import os
from collections import namedtuple

import numpy as np

_default_data_dir = os.path.join(os.getenv('HOME'), 'var/data/cifar')


def _unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def _to_onehot(k, a):
    vs = []
    for x in a:
        v = np.zeros((k, ), np.float32)
        v[x] = 1
        vs.append(v)
    return np.array(vs)


class Cifar10Loader(object):
    def __init__(self,
                 data_dir=_default_data_dir,
                 normalize=False,
                 one_hot=False):
        self._data_dir = data_dir
        self._normalize = normalize
        self._one_hot = one_hot

    def _load_batch(self, filename):
        x = _unpickle(filename)
        image_batch = x[b'data'].reshape(10000, 3, 32,
                                         32).transpose(0, 2, 3, 1)
        if self._normalize:
            image_batch = (image_batch / 255.0).astype(np.float32)
        label_batch = np.array(x[b'labels'])
        if self._one_hot:
            label_batch = _to_onehot(10, label_batch)
        return image_batch, label_batch

    def load_train(self):
        images = np.array([], np.uint8).reshape(0, 32, 32, 3)
        if self._one_hot:
            labels = np.array([], np.uint8).reshape(0, 10)
        else:
            labels = np.array([], np.uint8).reshape(0)
        for i in range(5):
            filename = os.path.join(self._data_dir, 'cifar-10-batches-py',
                                    'data_batch_%d' % (i + 1))
            image_batch, label_batch = self._load_batch(filename)
            images = np.concatenate((images, image_batch))
            labels = np.concatenate((labels, label_batch))
        return namedtuple('DataSet', 'images labels')(images, labels)

    def load_test(self):
        filename = os.path.join(self._data_dir, 'cifar-10-batches-py',
                                'test_batch')
        images, labels = self._load_batch(filename)
        return namedtuple('DataSet', 'images labels')(images, labels)

    def load_datasets(self):
        train = self.load_train()
        test = self.load_test()
        return namedtuple('Cifar10DataSets', 'train test')(train, test)


class Cifar100Loader(object):
    def __init__(self,
                 data_dir=_default_data_dir,
                 normalize=False,
                 one_hot=False):
        self._data_dir = data_dir
        self._normalize = normalize
        self._one_hot = one_hot

    def _load_batch(self, filename):
        x = _unpickle(filename)
        images = x[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        if self._normalize:
            images = (images / 255.0).astype(np.float32)
        labels = np.array(x[b'fine_labels'])
        if self._one_hot:
            labels = _to_onehot(100, labels)
        return images, labels

    def _load_dataset(self, name):
        filename = os.path.join(self._data_dir, 'cifar-100-python', name)
        images, labels = self._load_batch(filename)
        return namedtuple('DataSet', 'images labels')(images, labels)

    def load_train(self):
        return self._load_dataset('train')

    def load_test(self):
        return self._load_dataset('test')

    def load_datasets(self):
        train = self.load_train()
        test = self.load_test()
        return namedtuple('Cifar100DataSets', 'train test')(train, test)
