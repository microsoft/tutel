#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module.

Reference:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import os, sys

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION
except:
    IS_HIP_EXTENSION = False

if len(sys.argv) <= 1:
    sys.argv += ['install', '--user']

root_path = os.path.dirname(sys.argv[0])
root_path = root_path if root_path else '.'

os.chdir(root_path)

setup(
    name='tutel',
    version='0.1.0',
    description='An Optimized Mixture-of-Experts Implementation.',
    url='https://github.com/microsoft/Tutel',
    author='Microsoft',
    author_email='tutel@microsoft.com',
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: GPU',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=['Mixture of Experts', 'MoE', 'Optimization'],
    packages=find_packages(),
    python_requires='>=3.6, <4',
    install_requires=[
    ],
    ext_modules=[
        CUDAExtension('tutel_custom_kernel', [
            './tutel/custom/custom_kernel.cpp',
        ],
        library_dirs=['/usr/local/cuda/lib64/stubs'],
        libraries=['dl', 'cuda', 'nvrtc'] if not IS_HIP_EXTENSION else [])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    project_urls={
        'Source': 'https://github.com/microsoft/Tutel',
        'Tracker': 'https://github.com/microsoft/Tutel/issues',
    },
)
