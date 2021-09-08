# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module.

Reference:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='tutel',
    version='0.0.1',
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
    python_requires='>=3.5, <4',
    install_requires=[
        'ninja>=1.10.2',
    ],
    ext_modules=[
        CUDAExtension('tutel_custom_kernel', [
            './tutel/custom/custom_kernel.cpp',
        ],
        libraries=['dl'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    project_urls={
        'Source': 'https://github.com/microsoft/Tutel',
        'Tracker': 'https://github.com/microsoft/Tutel/issues',
    },
)
