import os
from collections import namedtuple

import numpy as np

from .idx import read_idx_file


def _to_onehot(k, a: np.ndarray):
    vs = []
    for x in a:
        v = np.zeros((k, ), np.float32)
        v[x] = 1
        vs.append(v)
    return np.array(vs)


def load_mnist_data(data_dir, prefix, normalize, one_hot):
    prefixes = ['train', 't10k']
    if not prefix in prefixes:
        raise ValueError('prefix must be %s' % ' | '.join(prefixes))
    image_file = os.path.join(data_dir, prefix + '-images-idx3-ubyte')
    label_file = os.path.join(data_dir, prefix + '-labels-idx1-ubyte')

    images = read_idx_file(image_file)
    if normalize:
        images = (images / 255.0).astype(np.float32)
    labels = read_idx_file(label_file)
    if one_hot:
        labels = _to_onehot(10, labels)

    return namedtuple('DataSet', 'images labels')(images, labels)


def load_datasets(data_dir, normalize=False, one_hot=False):
    train = load_mnist_data(data_dir, 'train', normalize, one_hot)
    test = load_mnist_data(data_dir, 't10k', normalize, one_hot)
    return namedtuple('MnistDataSets', 'train test')(train, test)
