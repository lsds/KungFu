"""
idx implements the IDX file format defined in
http://yann.lecun.com/exdb/mnist/
"""

import io
import sys
import tarfile
from struct import pack, unpack

import numpy as np

_idx_2_np = [
    (0x8, np.uint8),
    (0x9, np.int8),
    (0xb, np.int16),
    (0xc, np.int32),
    (0xd, np.float32),
    (0xe, np.float64),
]


def _get_idx_type(np_type):
    for idx_t, np_t in _idx_2_np:
        if np_t == np_type:
            return idx_t
    raise ValueError('unsupported np_type %s' % np_type)


def _get_np_type(idx_type):
    for idx_t, np_t in _idx_2_np:
        if idx_t == idx_type:
            return np_t
    raise ValueError('unsupported idx_type %s' % idx_type)


def write_idx_header(f, a):
    f.write(pack('BBBB', 0, 0, _get_idx_type(a.dtype), len(a.shape)))
    for dim in a.shape:
        # https://docs.python.org/3/library/struct.html#format-characters
        f.write(pack('>I', dim))


def write_idx_to(f, a):
    write_idx_header(f, a)
    f.write(a.tobytes())


def write_idx_file(name, a):
    with open(name, 'wb') as f:
        write_idx_to(f, a)


def read_idx_header(f):
    magic = f.read(4)  # [0, 0, dtype, rank]
    _, _, dtype, rank = magic
    if sys.version_info.major == 2:
        dtype = ord(dtype)
        rank = ord(rank)
    # https://docs.python.org/3/library/struct.html#format-characters
    dims = [unpack('>I', f.read(4))[0] for _ in range(rank)]
    return dtype, dims


def read_idx_from(f):
    dtype, dims = read_idx_header(f)
    return np.ndarray(dims, _get_np_type(dtype), f.read())


def read_idx_file(name):
    with open(name, 'rb') as f:
        return read_idx_from(f)


def _infer_out_filename(in_file):
    name = str(in_file)
    if name.endswith('.npz'):
        name = name[:-4]
    name += '.idx.tar'
    return name


def npz2idxtar(in_file, out_file=None):
    if out_file is None:
        out_file = _infer_out_filename(in_file)
    ws = np.load(in_file)
    with tarfile.open(out_file, 'w') as tar:
        for name in ws.files:
            w = ws[name]
            bs = io.BytesIO()
            write_idx_to(bs, w)
            info = tarfile.TarInfo(name)
            info.size = len(bs.getvalue())
            tar.addfile(info, io.BytesIO(bs.getbuffer()))
    return out_file
