"""distutils/setuptools extension."""

import os
import os.path
import sys
import distutils
import setuptools
import subprocess
import glob
import re
import shlex
from packaging import version as pversion
from distutils import ccompiler, unixccompiler
from distutils.command import build_ext as build_ext_base
from distutils.sysconfig import customize_compiler

_all__ = [
    'is_windows',
    'is_darwin',
    'cuda_home',
    'cuda_version',
    'cudnn_home',
    'cudnn_version',
    'build_ext',
]

def is_windows():
    return sys.platform == 'win32'


def is_darwin():
    return sys.platform.startswith('darwin')


def cuda_home():
    """Home of local CUDA."""
    # Guess #1
    home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if home is None:
        # Guess #2
        try:
            which = 'where' if is_windows() else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                               stderr=devnull).decode().rstrip('\r\n')
                home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if is_windows():
                homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(homes) == 0:
                    home = ''
                else:
                    home = homes[0]
            else:
                home = '/usr/local/cuda'
        if not os.path.exists(home):
            home = None
    if not home:
        print('-- CUDA not found.')
    return home


def cuda_version():
    """Version of local CUDA toolkit: (MAJOR, MINOR, PATCH)."""
    nvcc = os.path.join(cuda_home(), 'bin', 'nvcc')
    if not nvcc:
        return None
    with open(os.devnull, 'w') as devnull:
        version = subprocess.check_output([nvcc, '--version'], stderr=devnull).decode()
    match = re.search(r'V(?P<version>[0-9\.]+)$', version)
    version = pversion.parse(match.group('version')).release
    return version


def cudnn_home():
    """Home of local CuDNN."""
    home = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
    if home is None:
        home = cuda_home()
    if not os.path.exists(os.path.join(home, 'include', 'cudnn.h')):
        home = None
    if not home:
        print('-- CUDNN not found.')
    return home


def cudnn_version():
    """Version of local CuDNN: (MAJOR, MINOR, PATCH)."""
    def search_define(name, line, default=None):
        match = re.search(r'#\s*[Dd][Ee][Ff][Ii][Nn][Ee]\s+' + name +
                          r'\s+(?P<version>\d+)', line)
        if match:
            return int(match.group('version'))
        else:
            return default

    header = os.path.join(cudnn_home(), 'include', 'cudnn.h')
    with open(header, 'r') as file:
        lines = file.readlines()
    version = [None, None, None]
    for line in lines:
        if version[0] is None:
            version[0] = search_define('CUDNN_MAJOR', line, version[0])
        if version[1] is None:
            version[1] = search_define('CUDNN_MINOR', line, version[1])
        if version[2] is None:
            version[2] = search_define('CUDNN_PATCHLEVEL', line, version[2])
    return tuple(version)


def link_relative(path):
    if is_windows():
        return None
    elif is_darwin():
        return os.path.join('@loader_path', path)
    else:
        return os.path.join('$ORIGIN', path)

class SharedLibrary(setuptools.Extension):
    pass

def customize_compiler_for_dylib(self):
    def _link_dylib(objects, output_libname, *args, **kwargs):
        libname = self.library_filename(output_libname, lib_type='dylib')

        def add_install_name(extra_args, libname):
            has_install_name = any([a.startswith('-install_name')
                                    for a in extra_args])
            if not has_install_name:
                extra_args += ['-install_name',
                               os.path.join('@rpath', os.path.basename(libname))]
            return extra_args

        if len(args) > 4:
            args[3] = add_install_name(args[3], libname)
        else:
            kwargs['extra_postargs'] = add_install_name(kwargs.get('extra_postargs', []), libname)

        return self.link(ccompiler.CCompiler.SHARED_LIBRARY, objects,
                         libname, *args, **kwargs)
    self.link_shared_object = _link_dylib


def fix_compiler_rpath(self):
    if isinstance(self, unixccompiler.UnixCCompiler):
        func_original = self.runtime_library_dir_option

        def func_fixed(dir):
            if sys.platform[:6] == "darwin":
                return "-Wl,-rpath," + dir
                # return "-Xlinker -rpath -Xlinker " + dir
            else:
                return func_original(dir)

        self.runtime_library_dir_option = func_fixed


class build_ext(build_ext_base.build_ext):
    """Extends native extension builder to handle compiler flags."""

    def build_extension(self, ext):
        # Set proper compiler
        compiler0 = self.compiler
        self.compiler = make_compiler(compiler=self.compiler,
                                      language=ext.language,
                                      dry_run=self.dry_run,
                                      force=self.force)
        distutils.sysconfig.customize_compiler(self.compiler)
        fix_compiler_rpath(self.compiler)
        if self.include_dirs is not None:
            self.compiler.set_include_dirs(self.include_dirs)
        if self.define is not None:
            # 'define' option is a list of (name,value) tuples
            for (name, value) in self.define:
                self.compiler.define_macro(name, value)
        if self.undef is not None:
            for macro in self.undef:
                self.compiler.undefine_macro(macro)

        # Set proper linker
        if isinstance(ext, SharedLibrary):
            if sys.platform[:6] == "darwin":
                customize_compiler_for_dylib(self.compiler)
            else:
                self.compiler.link_shared_object = self.compiler.link_shared_lib

        # Set proper filename generator
        get_ext_filename0 = self.get_ext_filename
        if isinstance(ext, SharedLibrary):
            self.get_ext_filename = lambda x: x

        # OSX: change -bundle to -dynamiclib
        linker_so = []
        for arg in self.compiler.linker_so:
            linker_so += shlex.split(arg)
        linker_so = ['-dynamiclib' if arg == '-bundle' else arg
                     for arg in linker_so]
        self.compiler.set_executables(linker_so=linker_so)

        # Build extension
        super().build_extension(ext)

        # Reset
        self.compiler = compiler0
        self.get_ext_filename = get_ext_filename0

class NVCCompiler(unixccompiler.UnixCCompiler):
    compiler_type = 'nvcc'

    def __init__(self, verbose=0, dry_run=0, force=0):

        unixccompiler.UnixCCompiler.__init__(self, verbose, dry_run, force)

        self.set_executables(compiler='nvcc',
                             compiler_so='nvcc',
                             compiler_cxx='nvcc',
                             linker_exe='nvcc',
                             linker_so='nvcc -shared')
        self.src_extensions += ['.cu']


def get_compiler_type(compiler, plat=None):
    if plat is None:
        plat = os.name
    if isinstance(compiler, distutils.ccompiler.CCompiler):
        return compiler.compiler_type
    compiler = os.path.basename(str(compiler).lower())
    if compiler in ('unix', 'msvc', 'cygwin', 'mingw32', 'bcpp', 'nvcc'):
        return compiler
    if compiler in ('cc', 'c++', 'gcc', 'g++'):
        if plat == 'posix':
            return 'unix'
        return 'cygwin'
    if compiler in ('visualstudio', 'devstudio'):
        return 'msvc'
    if compiler in ('bcc32',):
        return 'bcpp'
    return None


compiler_class = {
    'unix':    ('distutils.unixccompiler', 'UnixCCompiler'),
    'msvc':    ('distutils._msvccompiler', 'MSVCCompiler'),
    'cygwin':  ('distutils.cygwinccompiler', 'CygwinCCompiler'),
    'mingw32': ('distutils.cygwinccompiler', 'Mingw32CCompiler'),
    'bcpp':    ('distutils.bcppcompiler', 'BCPPCompiler'),
    'nvcc':    ('buildtools', 'NVCCompiler'),
}


def make_compiler(plat=None, compiler=None, language=None,
                  verbose=0, dry_run=0, force=0):
    if plat is None:
        plat = os.name
    if compiler is None and language == 'cuda':
        compiler = 'nvcc'
    if compiler is None:
        compiler_type = ccompiler.get_default_compiler(plat)
    else:
        compiler_type = get_compiler_type(compiler, plat)

    try:
        (module_name, class_name) = compiler_class[compiler_type]
    except KeyError:
        msg = "don't know how to compile C/C++ code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler" % compiler
        raise distutils.error.DistutilsPlatformError(msg)

    try:
        __import__(module_name)
        module = sys.modules[module_name]
        klass = vars(module)[class_name]
    except ImportError:
        raise distutils.error.DistutilsModuleError(
              "can't compile C/C++ code: unable to load module '%s'" % \
              module_name)
    except KeyError:
        raise distutils.error.DistutilsModuleError(
               "can't compile C/C++ code: unable to find class '%s' "
               "in module '%s'" % (class_name, module_name))

    compiler_object = klass(None, dry_run, force)

    return compiler_object
