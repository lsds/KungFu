from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


def kungfu_tensorflow_lib():
    import tensorflow as tf
    return Extension(
        'kungfu.kungfu_tensorflow_ops',
        sources=[
            'src/negotiator.cpp',
        ],
        include_dirs=[
            tf.sysconfig.get_include(),
        ],
        libraries=[
            'tensorflow_framework',
        ],
        library_dirs=[
            tf.sysconfig.get_lib(),
        ],
        extra_compile_args=[
            '-std=c++11',
        ],
    )


setup(
    name='kungfu',
    version='0.0.0',
    packages=find_packages(),
    description='The ultimate distributed training framework for TensorFlow',
    url='https://github.com/luomai/kungfu',
    ext_modules=[
        kungfu_tensorflow_lib(),
    ],
    setup_requires=[],
    install_requires=[],
)
