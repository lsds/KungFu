import glob

from setuptools import find_packages, setup
from torch.utils import cpp_extension

package_dir = './srcs/python'


def kungfu_library_dir():
    # FIXME: assuming kungfu Tensorflow is installed
    from kungfu.loader import _module_path
    return _module_path()


setup(
    name='kungfu_torch',
    version='0.0.0',
    package_dir={
        '': package_dir,
    },
    packages=find_packages(package_dir),
    ext_modules=[
        cpp_extension.CppExtension(
            'kungfu_torch_ops',
            glob.glob('srcs/cpp/src/torch/ops/*.cpp') +
            glob.glob('srcs/cpp/src/torch/ops/cpu/*.cpp'),
            include_dirs=['./srcs/cpp/include'],
            library_dirs=[kungfu_library_dir()],
            libraries=[
                'kungfu',
                'kungfu_python',
            ],
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension,
    },
)
