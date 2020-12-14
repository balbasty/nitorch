import os
import re
import collections
import setup_cuda
import setup_utils
import setup_system
try:
    import torch
    torch_found = True
except ImportError:
    torch = None
    torch_found = False


def torch_version():
    version = torch.__version__.split('+')[0]
    return setup_utils.version.from_str(version)


def torch_cuda_version():
    return setup_utils.version.from_str(torch.version.cuda)


def torch_cudnn_version():
    return setup_utils.version.from_str(torch.backends.cudnn.version())


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


def torch_cxx_abi():
  return str(int(torch._C._GLIBCXX_USE_CXX11_ABI))


def torch_omp_lib():
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_library_dir = os.path.join(torch_dir, 'lib')
    if setup_system.is_darwin():
        libtorch = os.path.join(torch_library_dir, 'libtorch.dylib')
        linked_libs = os.popen('otool -L "{}"'.format(libtorch))
        if 'libiomp5' in linked_libs:
            return 'iomp5'
        elif 'libomp' in linked_libs:
            return 'omp'
        else:
            return None
    return None


def torch_libraries(with_cuda=False):
    version = torch_version()
    if version < setup_utils.version.from_str('1.5'):
        libraries = ['c10', 'torch', 'torch_python']
        if with_cuda:
            libraries += ['cudart', 'c10_cuda']
    else:
        libraries = ['c10', 'torch_cpu', 'torch_python', 'torch']
        if with_cuda:
            libraries += ['cudart', 'c10_cuda', 'torch_cuda']
    if not with_cuda and torch_parallel_backend() == 'AT_PARALLEL_OPENMP':
        libraries += omp_libraries()
    return libraries


def torch_library_dirs(with_cuda=False, cuda_home=None):
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_library_dir = os.path.join(torch_dir, 'lib')
    library_dirs = [torch_library_dir]
    if with_cuda:
        if cuda_home is None:
            cuda_home = setup_cuda.cuda_home(version=torch_cuda_version())
            library_dirs += setup_cuda.cuda_library_dirs(cuda_home)
    if not with_cuda and torch_parallel_backend() == 'AT_PARALLEL_OPENMP':
        library_dirs += omp_library_dirs()
    return library_dirs


def torch_include_dirs(with_cuda=False, cuda_home=None):
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_include_dir = os.path.join(torch_dir, 'include')
    include_dirs = [torch_include_dir,
                    os.path.join(torch_include_dir, 'torch', 'csrc', 'api', 'include'),
                    os.path.join(torch_include_dir, 'TH'),
                    os.path.join(torch_include_dir, 'THC')]
    if with_cuda:
        if cuda_home is None:
            cuda_home = setup_cuda.cuda_home(version=torch_cuda_version())
        cuda_include_dirs = os.path.join(cuda_home, 'include')
        include_dirs += [d for d in cuda_include_dirs if d != '/usr/include']
    if not with_cuda and torch_parallel_backend() == 'AT_PARALLEL_OPENMP':
        include_dirs += omp_include_dirs()
    return include_dirs


def torch_cuda_arch(cuda_home):

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

    # If not given, look into libtorch_cuda
    if not arch_list:
        cuobjdump = os.path.join(cuda_home(), 'bin', 'cuobjdump')
        torchdir = os.path.dirname(os.path.abspath(torch.__file__))
        libtorch = os.path.join(torchdir, 'lib')
        if is_windows():
            libtorch = os.path.join(libtorch, 'torch_cuda.lib')
        else:
            assert not is_darwin()
            libtorch = os.path.join(libtorch, 'libtorch_cuda.so')
        arch_list = os.popen(cuobjdump + " '" + libtorch + \
                             "' -lelf | awk -F. '{print $3}' | " \
                             "grep sm | sort -u").read().split('\n')
        print(arch_list)
        arch_list = [arch[3] + '.' + arch[4] for arch in arch_list if arch]
        ptx_list = os.popen(cuobjdump + " '" + libtorch + \
                             "' -lptx | awk -F. '{print $3}' | " \
                             "grep sm | sort -u").read().split('\n')
        ptx_list = [arch[3] + '.' + arch[4] for arch in ptx_list if arch]
        arch_list = [arch + '+PTX' if arch in ptx_list else arch
                     for arch in arch_list]
    elif arch_list == 'mine':
        #   this bit was in the torch extension util but I have replaced
        #   it with the bit above that looks into libtorch
        capability = torch.cuda.get_device_capability()
        arch_list = ['{}.{}'.format(capability[0], capability[1])]
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        arch_list = arch_list.replace(' ', ';')
        # Expand named arches
        for named_arch, archval in named_arches.items():
            arch_list = arch_list.replace(named_arch, archval)

        arch_list = arch_list.split(';')

    # sanity check
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError("Unknown CUDA arch ({}) or GPU not supported".format(arch))

    return arch_list
