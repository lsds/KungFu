import atexit
import json
import os
from ctypes import byref, c_char, c_char_p, c_int

from kungfu.loader import (_call_method, _call_method_with, _load_clib,
                           _module_path)

__all__ = [
    'current_cluster_size',
    'current_local_rank',
    'current_local_size',
    'current_rank',
    'detached',
    'run_barrier',
]


def _load_and_init_python_lib():
    _load_clib('libkungfu')
    _python_lib = _load_clib('libkungfu_python')
    if not os.getenv('KUNGFU_SINGLE_MACHINE_MULTIPROCESS'):
        _call_method(_python_lib, 'kungfu_python_init')
    has_nccl = _call_method(_python_lib, 'kungfu_python_init_nccl')
    return _python_lib, has_nccl


def _load_python_lib():
    _load_clib('libkungfu')
    _python_lib = _load_clib('libkungfu_python')
    return _python_lib


if os.getenv('KUNGFU_DISABLE_AUTO_LOAD'):
    _python_lib = _load_python_lib()
    _has_nccl = None
else:
    _python_lib, _has_nccl = _load_and_init_python_lib()


def init_from_config(config):
    global _has_nccl
    js = json.dumps(config)
    # https://stackoverflow.com/questions/61294630/ctypes-passing-a-·string-as-a-pointer-from-python-to-c
    _call_method_with(_python_lib, 'kungfu_python_init_from_json',
                      c_char_p(js.encode()))
    _has_nccl = _call_method(_python_lib, 'kungfu_python_init_nccl')


def _init_single_machine_multiple_process(rank, size):
    global _python_lib
    global _has_nccl
    _load_clib('libkungfu')
    _python_lib = _load_clib('libkungfu_python')
    _call_method_with(_python_lib, 'kungfu_python_init_single_machine', rank,
                      size)
    _has_nccl = _call_method(_python_lib, 'kungfu_python_init_nccl')


def _finalize_python_lib():
    if _has_nccl:
        _call_method(_python_lib, 'kungfu_python_finialize_nccl')
    _call_method(_python_lib, 'kungfu_python_finialize')


atexit.register(_finalize_python_lib)


def uid():
    """Get the uid of this peer."""
    return _python_lib.kungfu_uid()


def detached():
    """Check if the peer is detached."""
    return bool(_python_lib.kungfu_detached())


def current_rank():
    """Get the current rank of this peer."""
    return _python_lib.kungfu_rank()


def current_local_rank():
    """Get the current local rank of this peer."""
    return _python_lib.kungfu_local_rank()


def current_cluster_size():
    """Get the number of peers in the current cluster."""
    return _python_lib.kungfu_size()


def current_local_size():
    """Get the number of local peers in the current cluster."""
    return _python_lib.kungfu_local_size()


def _get_cuda_index():
    return _python_lib.kungfu_get_cuda_index()


def run_barrier():
    """Run the barrier operation eagerly."""
    _python_lib.kungfu_barrier()


def propose_new_size(new_size):
    # FIXME: check ctypes
    _python_lib.kungfu_propose_new_size(int(new_size))


def check_interference():
    return _python_lib.kungfu_check_interference()


def calc_stats():
    return _python_lib.kungfu_calc_stats()


def log_stats():
    return _python_lib.kungfu_log_stats()


def print_strategy_stats():
    return _python_lib.kungfu_print_strategy_stats()


def _get_other_ranks():
    self_rank = current_rank()
    ranks = list(range(current_cluster_size()))
    return [r for r in ranks if r != self_rank]


def show_cuda_version():
    if _has_nccl:
        _call_method(_python_lib, 'kungfu_show_cuda_version', force=True)
    else:
        print('NCCL is NOT enabled')


def show_nccl_version():
    if _has_nccl:
        _call_method(_python_lib, 'kungfu_show_nccl_version', force=True)
    else:
        print('NCCL is NOT enabled')


# unstable APIs


def _resize_from_url():
    changed = c_char()
    detached = c_char()
    _python_lib.kungfu_resize_from_url(byref(changed), byref(detached))
    return bool(ord(changed.value)), bool(ord(detached.value))


def resize(n=None):
    """Resize cluster with given new size, or from config server if n is None."""
    if n is None:
        return _resize_from_url()
    changed = c_char()
    detached = c_char()
    _python_lib.kungfu_resize(c_int(n), byref(changed), byref(detached))
    return bool(ord(changed.value)), bool(ord(detached.value))


def all_reduce_int_max(x):
    y = c_int(x)
    _python_lib.kungfu_all_reduce_int_max(byref(y))
    return int(y.value)
