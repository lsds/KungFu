from .loader import _call_method, _load_clib, _module_path


def _load_and_init_python_lib():
    _load_clib('libkungfu')
    _python_lib = _load_clib('libkungfu_python')
    _call_method(_python_lib, 'kungfu_python_init')
    has_gpu = _call_method(_python_lib, 'kungfu_python_init_gpu')
    return _python_lib, has_gpu


_python_lib, _has_gpu = _load_and_init_python_lib()


def current_rank():
    return _python_lib.kungfu_rank()


def current_cluster_size():
    return _python_lib.kungfu_cluster_size()


def run_barrier():
    _python_lib.kungfu_barrier()


def _get_other_ranks():
    self_rank = current_rank()
    ranks = list(range(current_cluster_size()))
    return [r for r in ranks if r != self_rank]
