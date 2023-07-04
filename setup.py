#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SETUP SCRIPT
============

Important environment variables:
CUDA_HOME
    Path to cuda toolkit (nvcc, etc.)
    By default, we try to find it in standard locations.
NI_COMPILED_BACKEND ({TS, C, MONAI}, default=TS)
    Which backend to use for our interpolation/solver functions
    By default, TS (TorchScript) is used. It does not require any compilation
    and is crossed-platform. "C" triggers the compilation of C++ (and
    optionally CUDA) routines . "MONAI" tries to use functions from the
    monai packages (which were ported from nitorch).
NI_USE_CUDA ({0, 1}, default=1)
    Whether to compile with cuda support.
    By default, yes if a correct CUDA toolkit is found.
TORCH_CUDA_ARCH_LIST ({all, mine, Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper}, default=mine)
    If "mine", we only compile for the architecture of the current GPU.
    If "all", compile for all architectures.
    If space-separated list of architectures, compile only for these archs.
NI_PYTORCH_TARGET
    I don't think think we use it, but I should check.
"""
from setuptools import setup, find_packages
from warnings import warn
import os
from configparser import ConfigParser
import versioneer

c_deprecation_warning = '''
Using a pure pytorch implementation by default! This differs from the 
prior behavior, where C++/CUDA extensions were always compiled.

To compile the C++/CUDA extensions, set the environment variable 
`NI_COMPILED_BACKEND="C"` prior to setup. To (try) to use MONAI's 
extensions, set `NI_COMPILED_BACKEND="MONAI"`. 

See the documentation for detail.
'''

SETUP_KWARGS = {}
CMDCLASS = versioneer.get_cmdclass()

config = ConfigParser()
rootdir = os.path.dirname(os.path.abspath(__file__))
config.read(os.path.join(rootdir, 'setup.cfg'))
INSTALL_REQUIRES = config['options']['install_requires']
INSTALL_REQUIRES = INSTALL_REQUIRES.split('\n')

COMPILED_BACKEND = os.environ.get('NI_COMPILED_BACKEND', '')
if not COMPILED_BACKEND:
    warn(c_deprecation_warning, DeprecationWarning)
    COMPILED_BACKEND = 'TS'

if COMPILED_BACKEND.upper() == 'C':
    from setup_cext import prepare_extensions
    from buildtools import build_ext
    from torch import __version__ as torch_version
    torch_version = torch_version.split('.')
    if '.'.join(torch_version[:2]) == '1.7':
        torch_version = '.'.join(torch_version[:3])  # we need the patch
    else:
        torch_version = '.'.join(torch_version[:3])
    PYTORCH_TARGET = os.environ.get('NI_PYTORCH_TARGET', '')
    SETUP_KWARGS['ext_package'] = 'nitorch'
    SETUP_KWARGS['ext_modules'] = prepare_extensions()
    CMDCLASS.update({'build_ext': build_ext})
    INSTALL_REQUIRES += [f'torch=={torch_version}']

if COMPILED_BACKEND.upper() == 'MONAI':
    INSTALL_REQUIRES += ['monai>=0.5']

SETUP_KWARGS['install_requires'] = INSTALL_REQUIRES
SETUP_KWARGS['cmdclass'] = CMDCLASS

setup(
    version=versioneer.get_version(),
    packages=find_packages(),
    **SETUP_KWARGS
)
