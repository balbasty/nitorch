#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:26:01 2020

@author: ybalba
"""

import os
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

SpatialCppExtension = CppExtension(
  name = 'C.cpu.spatial', 
  sources = ['nitorch/C/cpu/spatial.cpp', 'nitorch/C/pushpull.cpp'],
  define_macros = [('AT_PARALLEL_OPENMP', '1')],
  extra_compile_args = ['-fopenmp'],
)

# SpatialCUDAExtension = CUDAExtension(
#   name = 'C.cpu.spatial', 
#   sources = ['nitorch/C/cuda/spatial.cpp', 'nitorch/C/pushpull.cpp'],
# )

setup(name='nitorch',
      version='0.1a.dev',
      packages=find_packages(),
      install_requires=['torch>=1.5'],
      python_requires='>=3.0',
      setup_requires=['torch>=1.5'],
      ext_package='nitorch',
      ext_modules=[SpatialCppExtension],
      cmdclass={'build_ext': BuildExtension})
