import glob
import os

from setuptools import find_packages, setup
from torch.utils import cpp_extension

package_dir = './srcs/python'


def kungfu_library_dir():
    # FIXME: assuming kungfu Tensorflow is installed
    from kungfu.loader import _module_path
    return _module_path()


def find_cuda():
    # TODO: find cuda
    return '/usr/local/cuda'


def create_extension():
    srcs = []
    srcs += glob.glob('srcs/cpp/src/torch/common.cpp')
    srcs += glob.glob('srcs/cpp/src/torch/ops/cpu/*.cpp')

    include_dirs = [
        # FIXME: use tmp dir of pip
        os.path.join(os.path.dirname(__file__), './srcs/cpp/include')
    ]
    library_dirs = [kungfu_library_dir()]
    libraries = ['kungfu', 'kungfu_python']

    with_cuda = None
    import torch
    if torch.cuda.is_available():
        srcs += glob.glob('srcs/cpp/src/cuda/*.cpp')
        srcs += glob.glob('srcs/cpp/src/torch/ops/cuda/*.cpp')
        with_cuda = True
        include_dirs += [os.path.join(find_cuda(), 'include')]
        library_dirs += [os.path.join(find_cuda(), 'lib64')]
        libraries += ['cudart']
        srcs += ['srcs/cpp/src/torch/module_cuda.cpp']
    else:
        srcs += ['srcs/cpp/src/torch/module_cpu.cpp']

    return cpp_extension.CppExtension(
        'kungfu_torch_ops',
        srcs,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        with_cuda=with_cuda,
    )


setup(
    name='kungfu_torch',
    version='0.0.0',
    package_dir={
        '': package_dir,
    },
    packages=find_packages(package_dir),
    ext_modules=[
        create_extension(),
    ],
    cmdclass={
        # FIXME: parallel build, (pip_install took 1m16s)
        'build_ext': cpp_extension.BuildExtension,
    },
)
