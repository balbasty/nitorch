import glob
import os
import re
import subprocess
import sys
import setup_system
import setup_utils


def cuda_home(version=None):
    """Home of local CUDA."""

    if version:
        major = version.major()
        if major is None:
            major = '*'
        minor = version.minor()
        if minor is None:
            minor = '*'
    else:
        major = '*'
        minor = '*'

    # Guess #1: environment variable
    home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    home = check_cuda_home(home, version)

    if not home:
        # Guess #2: is nvcc on the path?
        try:
            which = 'where' if setup_system.is_win32() else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'], stderr=devnull)
            nvcc = nvcc.decode().rstrip('\r\n')
            home = os.path.dirname(os.path.dirname(nvcc))
            home = check_cuda_home(home, version)
        except Exception:
            pass

    if not home:
        # Guess #3: standard location
        if setup_system.is_windows():
            if setup_system.is_win32():
                home = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/'
            else:
                home = '/cygdrive/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/'
            homes = glob.glob(home + f'v{major}.{minor}')
        else:
            if version:
                homes = glob.glob(f'/usr/local/cuda-{major}.{minor}')
                homes += ['/usr/local/cuda']
            else:
                homes = ['/usr/local/cuda']
        for home in homes:
            home = check_cuda_home(home, version)
            if home:
                break

    if not home:
        home = None
        if version:
            print('-- CUDA not found.', file=sys.stderr)
        else:
            print(f'-- CUDA {major}.{minor} not found.', file=sys.stderr)
    else:
        found_version = cuda_version(home)
        print(f'-- Found CUDA {found_version}')

    return home


def cuda_version(home=None):
    """Version of local CUDA toolkit."""
    home = home or cuda_home()
    if not home:
        return None
    nvcc = os.path.join(home, 'bin', 'nvcc')
    if not os.path.exists(nvcc):
        return None
    with open(os.devnull, 'w') as devnull:
        version = subprocess.check_output([nvcc, '--version'], stderr=devnull)
        version = version.decode()
    match = None
    for line in version.split('\n'):
        match = re.search(r'V(?P<version>[0-9\.]+)$', line)
        if match:
            break
    if not match:
        print('-- Failed to parse cuda version', file=sys.stderr)
    version = setup_utils.version.from_str(match.group('version'))
    return version


def check_cuda_home(home, version=None):
    """Check that the path exists, contains nvcc and has the correct version.
    Returns None if path is not correct.
    """
    if not os.path.exists(home):
        return None
    if not os.path.exists(os.path.join(home, 'bin', 'nvcc')):
        return None
    if version:
        version_found = cuda_version(home)
        if not version_found:
            return None
        if version != version_found:
            return None
    return home


def cuda_library_dirs(home):
    home = home or cuda_home()
    if setup_system.is_windows():
        library_dirs = [os.path.join(home, 'lib/x64')]
    elif os.path.exists(os.path.join(home, 'lib64')):
        library_dirs = [os.path.join(home, 'lib64')]
    elif os.path.exists(os.path.join(home, 'lib')):
        library_dirs = [os.path.join(home, 'lib')]
    else:
        library_dirs = []
    return library_dirs


def cuda_include_dirs(home):
    home = home or cuda_home()
    return [os.path.join(home, 'include')]


# ---------------------------
# I don't use cudnn right now
# ---------------------------
#
# def cudnn_home():
#     """Home of local CuDNN."""
#     home = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
#     if home is None:
#         home = cuda_home()
#     if home and not os.path.exists(os.path.join(home, 'include', 'cudnn.h')):
#         home = None
#     if not home:
#         print('-- CUDNN not found.')
#     return home
#
#
# def cudnn_version():
#     """Version of local CuDNN: (MAJOR, MINOR, PATCH)."""
#     def search_define(name, line, default=None):
#         match = re.search(r'#\s*[Dd][Ee][Ff][Ii][Nn][Ee]\s+' + name +
#                           r'\s+(?P<version>\d+)', line)
#         if match:
#             return int(match.group('version'))
#         else:
#             return default
#
#     home = cudnn_home()
#     if not home:
#         return None
#     header = os.path.join(home, 'include', 'cudnn.h')
#     with open(header, 'r') as file:
#         lines = file.readlines()
#     version = [None, None, None]
#     for line in lines:
#         if version[0] is None:
#             version[0] = search_define('CUDNN_MAJOR', line, version[0])
#         if version[1] is None:
#             version[1] = search_define('CUDNN_MINOR', line, version[1])
#         if version[2] is None:
#             version[2] = search_define('CUDNN_PATCHLEVEL', line, version[2])
#     return tuple(version)
