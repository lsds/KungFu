from .loader import _call_method, _load_clib, _module_path


def _load_and_init_python_lib():
    _load_clib('libkungfu')
    _python_lib = _load_clib('libkungfu_python')
    _call_method(_python_lib, 'kungfu_python_init')
    has_gpu = _call_method(_python_lib, 'kungfu_python_init_gpu')
    return _python_lib, has_gpu


_python_lib, _has_gpu = _load_and_init_python_lib()


def current_rank():
    """Get the current rank of this peer."""
    return _python_lib.kungfu_rank()


def current_cluster_size():
    """Get the number of peers in the current cluster."""
    return _python_lib.kungfu_cluster_size()


def run_barrier():
    """Run the barrier operation eagerly."""
    _python_lib.kungfu_barrier()


def _get_other_ranks():
    self_rank = current_rank()
    ranks = list(range(current_cluster_size()))
    return [r for r in ranks if r != self_rank]


def show_nccl_version():
    if _has_gpu:
        method = 'kungfu_show_nccl_version'
        ok = _call_method(_python_lib, method)
        if not ok:
            raise RuntimeError('failed to call %s' % method)
    else:
        print('NCCL is NOT enabled')
