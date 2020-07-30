#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:26:01 2020

@author: ybalba
"""

# ~~~ imports
import os
import sys
import re
import subprocess
import glob
import packaging.version
import collections
import distutils
from setuptools import setup, command, find_packages, Extension
from buildtools import *
import torch


# ~~~ libnitorch files
# Note that sources are identical between libnitorch_cpu and libnitorch_cuda.
# The same code is compiled in two different ways to generate native and cuda
# code. This trick allows to minimize code duplication.
# Finally, libnitorch links to both these sub-libraries and dispatches
# according to the Tensor's device type.
libnitorch_cpu_sources = ['pushpull_common.cpp']
libnitorch_cuda_sources = ['pushpull_common.cpp']
libnitorch_sources = ['pushpull.cpp']
ext_spatial_sources = ['spatial.cpp']

# That's a bit ugly. TODO: use config files?
libnitorch_cpu_headers = ['common.h',
                          'interpolation.h', 'interpolation_common.h',
                          'bounds.h', 'bounds_common.h']
libnitorch_cuda_headers = ['common.h',
                           'interpolation.h', 'interpolation_common.h',
                           'bounds.h', 'bounds_common.h']
libnitorch_headers = ['pushpull_common.h', 'interpolation.h', 'bounds.h']
ext_spatial_headers = ['pushpull.h', 'interpolation.h', 'bounds.h']

# TODO
# . There is still quite a lot to do in setup and buildtools in order to make
#   things clean and work on multiple platforms.
# . I have to add abi checks and other smart tricks as in
#   torch.utils.cpp_extension

MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)

# ~~~ helpers
# Most of the helpers are in build tools. The remainign helpers defined here
# are specific to the version of pytorch that we compile against.

def torch_version(astuple=True):
    version = torch.__version__.split('.')
    version = tuple(int(v) for v in version)
    if len(version) == 2:
        version = version + (0,)
    if not astuple:
        version = version[0]*10000 + version[1]*100 + version[0]
    return version


def torch_cuda_version(astuple=True):
    version = torch.version.cuda.split('.')
    version = tuple(int(v) for v in version)
    if len(version) == 2:
        version = version + (0,)
    if not astuple:
        version = version[0]*10000 + version[1]*100 + version[0]
    return version


def torch_cudnn_version(astuple=True):
    version = torch.backends.cudnn.version()
    version = (version//1000, version//100 % 10, version % 100)
    if not astuple:
        version = version[0]*10000 + version[1]*100 + version[0]
    return version


def torch_parallel_backend():
    match = re.search('^ATen parallel backend: (?P<backend>.*)$',
                      torch._C._parallel_info(), re.MULTILINE)
    if match is None:
        return None
    backend = match.group('backend')
    if backend == 'OpenMP':
        return 'AT_PARALLEL_OPENMP'
    elif backend == 'native thread pool':
        return 'AT_PARALLEL_NATIVE'
    elif backend == 'native thread pool and TBB':
        return 'AT_PARALLEL_NATIVE_TBB'
    else:
        return None


def torch_abi():
  return str(int(torch._C._GLIBCXX_USE_CXX11_ABI))


def torch_libraries(use_cuda=False):
    version = torch_version(astuple=False)
    if version < 10500:
        libraries = ['c10', 'torch', 'torch_python']
        if use_cuda:
            libraries += ['cudart', 'c10_cuda']
    else:
        libraries = ['c10', 'torch_cpu', 'torch_python', 'torch']
        if use_cuda:
            libraries += ['cudart', 'c10_cuda', 'torch_cuda']
    return libraries


def torch_library_dirs(use_cuda=False, use_cudnn=False):
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_library_dir = os.path.join(torch_dir, 'lib')
    library_dirs = [torch_library_dir]
    if use_cuda:
        if is_windows():
            library_dirs += [os.path.join(cuda_home(), 'lib/x64')]
        elif os.path.exists(os.path.join(cuda_home(), 'lib64')):
            library_dirs += [os.path.join(cuda_home(), 'lib64')]
        elif os.path.exists(os.path.join(cuda_home(), 'lib')):
            library_dirs += [os.path.join(cuda_home(), 'lib')]
    if use_cudnn:
        if is_windows():
            library_dirs += [os.path.join(cudnn_home(), 'lib/x64')]
        elif os.path.exists(os.path.join(cudnn_home(), 'lib64')):
            library_dirs += [os.path.join(cudnn_home(), 'lib64')]
        elif os.path.exists(os.path.join(cudnn_home(), 'lib')):
            library_dirs += [os.path.join(cudnn_home(), 'lib')]
    return library_dirs


def torch_include_dirs(use_cuda=False, use_cudnn=False):
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_include_dir = os.path.join(torch_dir, 'include')
    include_dirs = [torch_include_dir,
                    os.path.join(torch_include_dir, 'torch', 'csrc', 'api', 'include'),
                    os.path.join(torch_include_dir, 'TH'),
                    os.path.join(torch_include_dir, 'THC')]
    if use_cuda:
        cuda_include_dir = os.path.join(cuda_home(), 'include')
        if cuda_include_dir != '/usr/include':
            include_dirs += [cuda_include_dir]
    if use_cudnn:
        include_dirs += [os.path.join(cudnn_home(), 'include')]
    return include_dirs


def cuda_check():
    local_version = cuda_version()
    torch_version = torch_cuda_version()
    ok = (local_version[0] == torch_version[0] and
          local_version[1] == torch_version[1])
    if not ok:
        print('Your version of CUDA is v{}.{} while PyTorch was compiled with'
              'CUDA v{}.{}. NiTorch cannot be compiled with CUDA.'.format(
              local_version[0], local_version[1],
              torch_version[0], torch_version[1]))
    return ok


def cudnn_check():
    local_version = cudnn_version()
    torch_version = torch_cudnn_version()
    ok = (local_version[0] == torch_version[0] and
          local_version[1] == torch_version[1])
    if not ok:
        print('Your version of CuDNN is v{}.{} while PyTorch was compiled with'
              'CuDNN v{}.{}. NiTorch cannot be compiled with CuDNN.'.format(
              local_version[0], local_version[1],
              torch_version[0], torch_version[1]))
    return ok


def cuda_arch_flags():
    """
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """

    # Note: keep combined names ("arch1+arch2") above single names, otherwise
    # string replacement may not do the right thing
    named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5']
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

    # The default is sm_30 for CUDA 9.x and 10.x
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)

    # If not given, determine what's needed for the GPU that can be found
    if not arch_list:
        capability = torch.cuda.get_device_capability()
        arch_list = ['{}.{}'.format(capability[0], capability[1])]
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        arch_list = arch_list.replace(' ', ';')
        # Expand named arches
        for named_arch, archval in named_arches.items():
            arch_list = arch_list.replace(named_arch, archval)

        arch_list = arch_list.split(';')

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError("Unknown CUDA arch ({}) or GPU not supported".format(arch))
        else:
            num = arch[0] + arch[2]
            flags.append('-gencode=arch=compute_{},code=sm_{}'.format(num, num))
            if arch.endswith('+PTX'):
                flags.append('-gencode=arch=compute_{},code=compute_{}'.format(num, num))

    return list(set(flags))


def torch_extension_flags(name):
    return ['-DTORCH_EXTENSION_NAME={}'.format(name),
            '-DTORCH_API_INCLUDE_EXTENSION_H']


def gcc_clang_flags():
    return ['-fPIC', '-std=c++14']


def msvc_flags():
    return ['/MD', '/wd4819', '/EHsc']


def nvcc_flags():
    return [
      '-x=cu',
      '-D__CUDA_NO_HALF_OPERATORS__',
      '-D__CUDA_NO_HALF_CONVERSIONS__',
      '-D__CUDA_NO_HALF2_OPERATORS__',
      '--expt-relaxed-constexpr']


def omp_flags():
    if is_windows():
        return ['/openmp']
    elif is_darwin():
        # https://stackoverflow.com/questions/37362414/
        return ['-fopenmp=libiomp5']
    else:
        return ['-fopenmp']


def common_flags():
    if is_windows():
        return msvc_flags()
    else:
        return gcc_clang_flags()


def torch_flags(cuda=False):
    version = torch_version()
    version = version[0]*10000+version[1]*100+version[2]
    flags = ['-DNI_TORCH_VERSION=' + str(version)]
    backend = torch_parallel_backend();
    flags += [
        '-D' + torch_parallel_backend() + '=1',
        '-D_GLIBCXX_USE_CXX11_ABI=' + torch_abi()]
    if not cuda and backend == 'AT_PARALLEL_OPENMP':
        flags += omp_flags()
    return flags


def torch_link_flags(cuda=False):
    backend = torch_parallel_backend();
    flags = []
    if not cuda and backend == 'AT_PARALLEL_OPENMP':
        flags += omp_flags()
    return flags


def cuda_flags():
    flags = nvcc_flags() + cuda_arch_flags()
    if is_windows():
        for flag in common_flags():
            flags = ['-Xcompiler', flag] + flags
    else:
        for flag in common_flags():
            flags += ['--compiler-options', flag]
    return flags


def abspathC(files):
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    sourcedir = os.path.join(scriptdir, 'nitorch', '_C')
    return [os.path.join(sourcedir, f) for f in files]

# ~~~ checks
use_cuda = cuda_home() and cuda_check()
use_cudnn = False  # cudnn_home() and cudnn_check()


build_extensions = []
nitorch_lib = []
nitorch_libext = []
# ~~~ setup libraries
NiTorchCPULibrary = SharedLibrary(
    name='lib.nitorch_cpu',
    sources=abspathC(libnitorch_cpu_sources),
    depends=abspathC(libnitorch_cpu_headers),
    libraries=torch_libraries(),
    library_dirs=torch_library_dirs(),
    include_dirs=torch_include_dirs(),
    extra_compile_args=common_flags() + torch_flags(),
    extra_link_args=torch_link_flags(),
    language='c++',
)
build_extensions += [NiTorchCPULibrary]
nitorch_libext += [NiTorchCPULibrary]
nitorch_lib += ['nitorch_cpu']
if use_cuda:
    NiTorchCUDALibrary = SharedLibrary(
        name='lib.nitorch_cuda',
        sources=abspathC(libnitorch_cuda_sources),
        depends=abspathC(libnitorch_cuda_headers),
        libraries=torch_libraries(use_cuda),
        library_dirs=torch_library_dirs(use_cuda, use_cudnn),
        include_dirs=torch_include_dirs(use_cuda, use_cudnn),
        extra_compile_args=cuda_flags() + torch_flags(cuda=True),
        extra_link_args=torch_link_flags(cuda=True),
        language='cuda',
    )
    build_extensions += [NiTorchCUDALibrary]
    nitorch_libext += [NiTorchCUDALibrary]
    nitorch_lib += ['nitorch_cuda']
NiTorchLibrary = SharedLibrary(
    name='lib.nitorch',
    sources=abspathC(libnitorch_sources),
    depends=nitorch_libext + abspathC(libnitorch_headers),
    libraries=torch_libraries() + nitorch_lib,
    library_dirs=torch_library_dirs(),
    include_dirs=torch_include_dirs(),
    extra_compile_args=common_flags() + torch_flags() + (['-DNI_WITH_CUDA'] if use_cuda else []),
    runtime_library_dirs=[link_relative('.')],
    language='c++',
)
build_extensions += [NiTorchLibrary]
nitorch_libext = [NiTorchLibrary]
nitorch_lib = ['nitorch']
# ~~~ setup extensions
SpatialExtension = Extension(
    name='_C.spatial',
    sources=abspathC(ext_spatial_sources),
    depends=nitorch_libext + abspathC(ext_spatial_headers),
    libraries=torch_libraries(use_cuda) + nitorch_lib,
    library_dirs=torch_library_dirs(use_cuda, use_cudnn),
    include_dirs=torch_include_dirs(use_cuda, use_cudnn),
    extra_compile_args=common_flags() + torch_flags() + torch_extension_flags('spatial'),
    runtime_library_dirs=[link_relative(os.path.join('..', 'lib'))]
)
build_extensions += [SpatialExtension]


setup(
    name='nitorch',
    version='0.1a.dev',
    packages=find_packages(),
    install_requires=['torch>=1.5'],
    python_requires='>=3.0',
    setup_requires=['torch>=1.5'],
    ext_package='nitorch',
    ext_modules=build_extensions,
    cmdclass={'build_ext': build_ext}
)
