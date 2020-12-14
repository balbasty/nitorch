import os
import warnings
import sys
import setup_compiler
import setup_cuda
import setup_utils
import setup_system
import setup_torch


class BuildConfig:

    platform = None
    os_type = None
    compiler = None
    compiler_version = None
    cuda_home = None
    cuda_version = None
    cuda_arch = None
    cudnn_version = None
    torch_version = None
    torch_parallel_backend = None
    torch_cxx_abi = None
    torch_omp = None
    torch_libraries = None
    torch_library_dirs = None
    torch_include_dirs = None

    def __init__(self):
        # the order in which these are called matters
        self.set_system()
        self.set_torch()
        self.set_cuda()
        self.finalize()

    def set_system(self):
        self.platform = setup_system.guess_platform()
        self.os_type = setup_system.guess_os_type()
        print(f'-- System: {self.platform} ({self.os_type})')

    def set_compiler(self):
        self.compiler = setup_compiler.guess_compiler_name()
        self.compiler_version = setup_compiler.guess_compiler_version()
        print(f'-- C++ compiler: {self.compiler} {self.compiler_version}')

    def set_torch(self):
        if not setup_torch.torch_found:
            print('-- PyTorch not found', file=sys.stderr)
            raise RuntimeError('Failed to import torch.')
        self.torch_version = setup_torch.torch_version()
        self.cuda_version = setup_torch.torch_cuda_version()
        if self.cuda_version:
            self.cuda_arch = setup_torch.arc
        self.torch_parallel_backend = setup_torch.torch_parallel_backend()
        self.torch_cxx_abi = setup_torch.torch_cxx_abi()
        self.torch_omp = setup_torch.torch_omp_lib()
        print(f'-- Torch: version {self.torch_version}')
        if self.cuda_version:
            print(f'--      CUDA: {self.cuda_version}')
        else:
            print(f'--      CUDA: No')
        print(f'--      Parallel backend: {self.torch_parallel_backend}')
        if self.torch_parallel_backend == 'AT_PARALLEL_OPENMP':
            print(f'--      OpenMP library: {self.torch_omp}')
        print(f'--      CXX ABI: {self.torch_cxx_abi}')

    def set_cuda(self):
        if self.cuda_version is None:
            # Torch was not compiled with cuda
            return
        cuda_home = setup_cuda.cuda_home(version=self.cuda_version)
        if not cuda_home:
            print(f'-- CUDA {self.cuda_version} not found', file=sys.stderr)
            print(f'--      Compiling nitorch without CUDA', file=sys.stderr)




