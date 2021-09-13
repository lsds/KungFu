import os
from kungfu.python import _python_lib
from kungfu.python import current_rank, current_cluster_size

from ctypes import byref, c_char, c_int, c_char_p


def read_shard_result(filename):
    lines = [line.strip() for line in open(filename)]
    n = int(lines[0])
    filenames = [lines[1:n + 1]]
    m = int(lines[n + 1])
    batch_sizes = []
    for i in range(m):
        bs, count = lines[n + 2 + i].split(' ')
        batch_sizes.append((int(bs), int(count)))
    return {
        'filenames': filenames,
        'batch_sizes': batch_sizes,
    }


def create_tf_records(index_file, seed, global_batch_size):
    rank = current_rank()
    size = current_cluster_size()
    print('create_tf_records for peer(%d/%d)' % (rank, size))

    #  FIXME: return filenames
    _python_lib.kungfu_create_tf_records(c_char_p(index_file.encode()),
                                         c_int(seed), c_int(global_batch_size))

    progress = int(os.getenv('KUNGFU_INIT_PROGRESS'))
    output = "tf-files-from-%d-%d-of-%d.list.txt" % (progress, rank, size)

    shard = read_shard_result(output)

    return shard
