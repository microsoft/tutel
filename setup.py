#!/usr/bin/env python3

import os, shutil, sys

if len(sys.argv) <= 1:
    sys.argv += ['install', '--user']

root_path = os.path.dirname(sys.argv[0])
os.chdir(root_path if root_path else '.')
root_path = '.'

os.chdir('./tutel/custom')

sys.dont_write_bytecode = False

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cpp_flags = ['-w']

setup(
    name='tutel_custom_kernel',
    ext_modules=[
        CUDAExtension('tutel_custom_kernel', [
            'custom_kernel.cpp',
        ],
		extra_compile_args={'cxx': cpp_flags, 'nvcc': cpp_flags},
        libraries=[])
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
  path = '%s/%s' % (site.USER_SITE, site_name)
  try:
    shutil.rmtree(path)
  except:
    pass
  shutil.copytree(dir_name, path)

user_setup('../../tutel', 'tutel_moe')
user_setup('../../baseline', 'baseline_moe')
