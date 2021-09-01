#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from warnings import warn
import os

c_deprecation_warning = '''
Trying to compile C extensions by default (for backward
compatibility). If you wish to skip compilation and use a pure 
PyTorch implementation instead, set the environment variable 
NI_COMPILED_BACKEND="TS" before setup. 
 
We will soon use the pure PyTorch version by default. When it 
is the case, set NI_COMPILED_BACKEND="C" to compile the C 
extensions.
'''

COMPILED_BACKEND = os.environ.get('NI_COMPILED_BACKEND', '')
if not COMPILED_BACKEND:
    warn(c_deprecation_warning, DeprecationWarning)
    COMPILED_BACKEND = 'C'

if COMPILED_BACKEND.upper() == 'C':
    from setup_cext import prepare_extensions
    from buildtools import build_ext
    cext_cmd = {'build_ext': build_ext}
    cext_modules = prepare_extensions()
else:
    cext_cmd = {}
    cext_modules = []


INSTALL_REQUIRES = [
    'torch>=1.5',
    'wget', 'appdirs',  # < used for downloading nitorch data
    'numpy', 'scipy',   # < used only in spm/affine_reg
]
if COMPILED_BACKEND.upper() == 'MONAI':
    INSTALL_REQUIRES += ['monai>=0.5']

setup(
    name='nitorch',
    version='0.1a.dev',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.6',
    ext_package='nitorch',
    ext_modules=cext_modules,
    cmdclass=cext_cmd,
    entry_points={'console_scripts': ['nitorch=nitorch.cli:cli']}
)
