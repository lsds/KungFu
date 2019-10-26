import os
from collections import namedtuple

import numpy as np

from .idx import read_idx_file


def _to_onehot(k, a):
    vs = []
    for x in a:
        v = np.zeros((k, ), np.float32)
        v[x] = 1
        vs.append(v)
    return np.array(vs)


def load_mnist_data(data_dir, prefix, normalize, one_hot, padded=False):
    prefixes = ['train', 't10k']
    if not prefix in prefixes:
        raise ValueError('prefix must be %s' % ' | '.join(prefixes))
    image_file = os.path.join(data_dir, prefix + '-images-idx3-ubyte')
    label_file = os.path.join(data_dir, prefix + '-labels-idx1-ubyte')

    images = read_idx_file(image_file)

    if padded:
        images = images.reshape((images.shape[0], 28, 28, 1))
        # Pad images with 0s
        images = np.pad(images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    if normalize:
        images = (images / 255.0).astype(np.float32)
    labels = read_idx_file(label_file)
    if one_hot:
        labels = _to_onehot(10, labels)

    return namedtuple('DataSet', 'images labels')(images, labels)


def load_datasets(data_dir, normalize=False, one_hot=False, padded=False):
    train = load_mnist_data(data_dir,
                            'train',
                            normalize,
                            one_hot,
                            padded=padded)
    test = load_mnist_data(data_dir, 't10k', normalize, one_hot, padded=padded)
    return namedtuple('MnistDataSets', 'train test')(train, test)
