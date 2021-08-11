#!/usr/bin/env python3

import os, sys

if len(sys.argv) <= 1:
    sys.argv += ['install']

root_path = os.path.dirname(sys.argv[0])
os.chdir(root_path if root_path else '.')
root_path = '.'

sys.dont_write_bytecode = False

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cpp_flags = ['-w']

setup(
    name='custom_kernel',
    ext_modules=[
        CUDAExtension('custom_kernel', [
            'custom_kernel_cuda.cpp',
        ], extra_compile_args={'cxx': cpp_flags, 'nvcc': cpp_flags},
        libraries=['cuda',])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

