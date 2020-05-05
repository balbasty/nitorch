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

setup(name='nitorch',
      version='0.1a.dev',
      packages=find_packages(),
      install_requires=['torch>=1.5'],
      python_requires='>=3.0',
      setup_requires=['torch>=1.5'],
      ext_package='nitorch',
      ext_modules=[CppExtension('C.spatial', ['nitorch/C/spatial.cpp']),
                  ],
      cmdclass={'build_ext': BuildExtension})
