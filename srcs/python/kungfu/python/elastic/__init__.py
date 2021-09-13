from kungfu.python import _python_lib

from ctypes import byref, c_char, c_int, c_char_p


def create_tf_records(index_file, seed, global_batch_size):
    print('create_tf_records')
    _python_lib.kungfu_create_tf_records(c_char_p(index_file.encode()),
                                         c_int(seed), c_int(global_batch_size))
