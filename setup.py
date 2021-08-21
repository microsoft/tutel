#!/usr/bin/env python3

import os, shutil, sys

if len(sys.argv) <= 1:
    sys.argv += ['install']

root_path = os.path.dirname(sys.argv[0])
os.chdir(root_path if root_path else '.')
root_path = '.'

os.chdir('./tutel/custom')

sys.dont_write_bytecode = False

'''
for tree in ('custom_kernel.egg-info', 'build', 'dist'):
  try:
    shutil.rmtree(f'{root_path}/{tree}')
  except:
    pass
'''

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
	from torch.utils.cpp_extension import IS_HIP_EXTENSION
except:
	IS_HIP_EXTENSION = False

cpp_flags = ['-w']

setup(
    name='custom_kernel',
    ext_modules=[
        CUDAExtension('custom_kernel', [
            'custom_kernel_cuda.cpp',
        ],
		extra_compile_args={'cxx': cpp_flags, 'nvcc': cpp_flags},
        libraries=['cuda'] if not IS_HIP_EXTENSION else [])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


# Copy library to user-site directly
import site

try:
  os.makedirs(site.USER_SITE)
except FileExistsError:
  pass

def user_setup(dir_name, site_name):
  path = f'{site.USER_SITE}/{site_name}'
  try:
    shutil.rmtree(path)
  except:
    pass
  shutil.copytree(dir_name, path)

user_setup('../../tutel', 'tutel_moe')
user_setup('../../baseline', 'baseline_moe')
