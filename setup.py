import os
import subprocess
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, sourcedir):
        Extension.__init__(self, '', sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def ensure_absent(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)


def cmake_flag(k, v):
    return '-D%s=%s' % (k, str(v))


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        extdir = self.get_ext_fullpath(ext.name)
        if not os.path.exists(extdir):
            os.makedirs(extdir)

        install_prefix = os.path.abspath(os.path.dirname(extdir))
        executable_dir = os.path.abspath(os.path.dirname(sys.executable))

        cmake_args = [
            # FIXME: use CMAKE_LIBRARY_OUTPUT_DIRECTORY
            cmake_flag('LIBRARY_OUTPUT_PATH',
                       os.path.join(install_prefix, 'kungfu')),
            cmake_flag('KUNGFU_BUILD_TF_OPS', 1),
            cmake_flag('KUNGFU_BUILD_TOOLS', 1),
            cmake_flag('CMAKE_RUNTIME_OUTPUT_DIRECTORY', executable_dir),
            cmake_flag('PYTHON', sys.executable),

            # uncomment to debug
            # cmake_flag('CMAKE_VERBOSE_MAKEFILE', 1),
            # cmake_flag('CMAKE_EXPORT_COMPILE_COMMANDS', 1),
        ]

        use_nccl = os.getenv('KUNGFU_USE_NCCL')
        if use_nccl:
            cmake_args.append(cmake_flag('KUNGFU_USE_NCCL', use_nccl))
            nccl_home = os.getenv('NCCL_HOME')
            if nccl_home:
                cmake_args.append(cmake_flag('NCCL_HOME', nccl_home))

        ensure_absent(os.path.join(ext.sourcedir, 'CMakeCache.txt'))

        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
        )
        subprocess.check_call(
            [
                'cmake',
                '--build',
                '.',
            ],
            cwd=self.build_temp,
        )


package_dir = './srcs/python'

setup(
    name='kungfu',
    version='0.0.0',
    package_dir={'': package_dir},
    packages=find_packages(package_dir),
    description='KungFu distributed machine learning framework',
    url='https://github.com/lsds/KungFu',
    ext_modules=[
        CMakeExtension('.'),
    ],
    cmdclass=dict(build_ext=CMakeBuild),
    setup_requires=[],
    install_requires=[],
)
