import glob

from setuptools import find_packages, setup
from torch.utils import cpp_extension

package_dir = './srcs/python'

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
            glob.glob('srcs/cpp/src/torch/ops/cpu/*.cpp'),
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension,
    },
)
