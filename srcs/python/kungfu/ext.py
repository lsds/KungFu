import atexit

from .loader import _call_method, _load_clib, _module_path


def _load_and_init_python_lib():
    _load_clib('libkungfu')
    _python_lib = _load_clib('libkungfu_python')
    _call_method(_python_lib, 'kungfu_python_init')
    has_nccl = _call_method(_python_lib, 'kungfu_python_init_nccl')
    return _python_lib, has_nccl


_python_lib, _has_nccl = _load_and_init_python_lib()


def _finalize_python_lib():
    if _has_nccl:
        _call_method(_python_lib, 'kungfu_python_finialize_nccl')
    _call_method(_python_lib, 'kungfu_python_finialize')


atexit.register(_finalize_python_lib)


def current_rank():
    """Get the current rank of this peer."""
    return _python_lib.kungfu_rank()


def current_local_rank():
    return _python_lib.kungfu_local_rank()


def current_cluster_size():
    """Get the number of peers in the current cluster."""
    return _python_lib.kungfu_cluster_size()


def _get_cuda_index():
    return _python_lib.kungfu_get_cuda_index()


def run_barrier():
    """Run the barrier operation eagerly."""
    _python_lib.kungfu_barrier()


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
