"""distutils/setuptools extension."""

# python
import os
import os.path
import sys
import subprocess
import glob
import re
import shlex

# (from sklearn) import setuptools before because it monkey-patches distutils
from setuptools import Extension as stExtension
import distutils
from distutils import ccompiler, unixccompiler
from distutils.command import build_ext as build_ext_base
from distutils.sysconfig import customize_compiler
from distutils.extension import Extension as duExtension

_all__ = [
    'is_windows',
    'is_darwin',
    'cuda_home',
    'cuda_version',
    'cudnn_home',
    'cudnn_version',
    'build_ext',
]

# ======================================================================
# We start with a few exported utilities that inform about the
# current system (os, location of the cuda install, ...)
# ======================================================================

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
    match = None
    for line in version.split('\n'):
        match = re.search(r'V(?P<version>[0-9\.]+)$', line)
        if match:
            break
    if not match:
        raise RuntimeError('Failed to parse cuda version')
    version = match.group('version').split('.')
    version = tuple(int(v) for v in version)
    return version


def cudnn_home():
    """Home of local CuDNN."""
    home = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
    if home is None:
        home = cuda_home()
    if home and not os.path.exists(os.path.join(home, 'include', 'cudnn.h')):
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

    home = cudnn_home()
    if not home:
        return None
    header = os.path.join(home, 'include', 'cudnn.h')
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


# ======================================================================
# Here, we monkey-patch distutils to:
# 1) Compile "shared library extensions", which are not python extensions
#    but dynamic libraries that python extensions will link against.
#    For this, we define a `SharedLibrary` extension class and write our
#    a specialized `build_ext` that knows about it.
# 2) Compile cuda code.
#    For this, we define and register a new `NVCCompiler` class.
# 3) Parallelize compilation at the .cpp level.
#    Python is used to deal with extensions that are made of a single
#    (or a few) source, so parallelize at the extension level. We're
#    starting to get a bunch of source files per extension, so we
#    parallelize at the source level instead.
#    To do this, we monkey-patch the `compile` method of the compilers.
#    Currently, we always use as many workers as cpu cores. Maybe we should
#    hack/add the -j option somehow so that it can be user-defined.
# ======================================================================

def link_relative(path):
    # find the correct way to set a relative path to a shared library
    if is_windows():
        return None
    elif is_darwin():
        return os.path.join('@loader_path', path)
    else:
        return os.path.join('$ORIGIN', path)


class SharedLibrary(stExtension):
    """Special extension for dynamic shared libraries.

    This special type of Extension is not a Python extension but
    a dynamic shared library against which other Python extensions
    (or other shared libraries) can link. The output type of these
    libraries is .so on Linux, .dylib on MacOS and ??? on Windows.
    """
    # We don't actually implement anything specific.
    # We just want build_ext to switch based on the extension type.
    pass


def customize_compiler_for_shared_lib(self):
    # Monkey patch compilers when they are compiling a shared lib.
    # - on MacOS, we need the lib to find its dependencies in a relative
    #   manner.

    def _link(objects, libpath, *args, **kwargs):
        def add_install_name(extra_args, libpath):
            has_install_name = any([a.startswith('-install_name')
                                    for a in extra_args])
            if not has_install_name:
                extra_args += ['-install_name',
                               os.path.join('@rpath', os.path.basename(libpath))]
            return extra_args

        # On MacOS: we need to manually specify a name stored inside the dylib
        if is_darwin():
            args = list(args)
            if len(args) > 4:
                args[3] = add_install_name(args[3], libpath)
            else:
                kwargs['extra_postargs'] = add_install_name(kwargs.get('extra_postargs', []), libpath)

        return self.link(ccompiler.CCompiler.SHARED_LIBRARY, objects,
                         libpath, *args, **kwargs)
    self.link_shared_object = _link


def fix_compiler_rpath(self):
    # On MacOS, the rpath option is a bit different
    if isinstance(self, unixccompiler.UnixCCompiler):
        func_original = self.runtime_library_dir_option

        def func_fixed(dir):
            if sys.platform[:6] == "darwin":
                return "-Wl,-rpath," + dir
                # return "-Xlinker -rpath -Xlinker " + dir
            else:
                return func_original(dir)

        self.runtime_library_dir_option = func_fixed


def fix_compile_parallel(self):
    # Patch `compile` so that it parallelizes compilation of cpp sources
    def compile_parallel(
            sources, output_dir=None, macros=None,
            include_dirs=None, debug=0, extra_preargs=None,
            extra_postargs=None, depends=None):
        macros, objects, extra_postargs, pp_opts, build = \
                self._setup_compile(output_dir, macros, include_dirs, sources,
                                    depends, extra_postargs)
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

        workers = int(os.environ.get('NI_COMPILE_WORKERS', '0'))
        workers = workers or os.cpu_count()  # may be None
        try:
            from concurrent.futures import ThreadPoolExecutor
        except ImportError:
            workers = None

        def compile1(obj):
            try:
                src, ext = build[obj]
            except KeyError:
                return
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        if workers is None:
            for obj in objects:
                compile1(obj)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(compile1, obj) for obj in objects]
                for fut in futures:
                    fut.result()

        # Return *all* object filenames, not just the ones we just built.
        return objects

    self.compile = compile_parallel


class build_ext(build_ext_base.build_ext):
    """Extends native extension builder.
    This class handles:
        . shared libraries (see the SharedLibrary class)
        . cuda extensions
        . dependencies between extensions
    """

    def _build_dependency_graph(self, extensions):
        # Build layers of stuff that can be compiled in parallel
        from copy import copy
        compiled = []  # Stuff that already found its layer
        uncompiled = copy(extensions)  # Stuff that is not in a layer yet
        layers = []  # Actual layers
        while len(uncompiled) > 0:
            nbuncompiled = len(uncompiled)  # Used to detect deadlocks
            layer = []
            uncompiled0 = []
            for ext in uncompiled:
                depends = [dep for dep in ext.depends
                           if isinstance(dep, duExtension)]
                if all([dep in compiled for dep in depends]):
                    layer.append(ext)
                else:
                    uncompiled0.append(ext)
            layers.append(layer)
            compiled += layer
            uncompiled = uncompiled0
            if len(uncompiled) == nbuncompiled:
                raise RuntimeError('Deadlock detected')
        return layers

    def _build_extensions_parallel(self):
        # We reimplement this one to take dependencies into account.
        workers = self.parallel
        if self.parallel is True:
            workers = os.cpu_count()  # may return None
        try:
            from concurrent.futures import ThreadPoolExecutor
        except ImportError:
            workers = None

        if workers is None:
            self._build_extensions_serial()
            return

        layers = self._build_dependency_graph(self.extensions)
        for layer in layers:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self.build_extension, ext)
                           for ext in layer]
                for ext, fut in zip(layer, futures):
                    with self._filter_build_errors(ext):
                        fut.result()

    def get_ext_filename_shared_lib(self, ext_name):
        r"""Convert the name of a SharedLibrary extension (eg. "foo.bar")
        into the name of the file from which it will be loaded (eg.
        "foo/libbar.so", or "foo/libbar.dylib").
        """
        if sys.platform[:6] == "darwin":
            lib_type = 'dylib'
        else:
            lib_type = 'shared'
        return self.compiler.library_filename(ext_name, lib_type=lib_type)

    def _get_ext_fullpath(self, ext):
        get_ext_filename0 = self.get_ext_filename
        if isinstance(ext, SharedLibrary):
            self.get_ext_filename = self.get_ext_filename_shared_lib
        output_path = self.get_ext_fullpath(ext.name)
        self.get_ext_filename = get_ext_filename0
        return output_path

    def _get_ext_dir(self, ext):
        return os.path.dirname(self.get_ext_fullpath(ext.name))

    def get_libraries(self, ext):
        """SharedLibrary extensions do not link against python."""
        if isinstance(ext, SharedLibrary):
            return ext.libraries
        else:
            return super().get_libraries(ext)

    def build_extension(self, ext):
        """Select appropriate compiler and linker.

        . CUDA          -> NVCCompiler
        . SharedLibrary -> link_shared_lib + don't append python version
        . MacOS         -> dylib + -dynamiclib + rpath
        . Depends       -> add appropriate -L
        """
        # Make temporary build directory extension dependent
        # (we compile the same sources with different compilers/flags
        #  to generate libnitorch_cpu and libnitorch_cuda)
        build_temp0 = self.build_temp
        self.build_temp = os.path.join(
            build_temp0, os.path.join(*ext.name.split('.')))

        # Select appropriate compiler for the extensions's language
        compiler0 = self.compiler
        self.compiler = make_compiler(language=ext.language,
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
            customize_compiler_for_shared_lib(self.compiler)

        # Add library dirs for dependencies
        for dep in ext.depends:
            if isinstance(dep, duExtension):
                self.compiler.add_library_dir(self._get_ext_dir(dep))

        # Use output filepath instead of Extension (for newer_group)
        depends = ext.depends
        ext.depends = [dep if not isinstance(dep, duExtension)
                       else self._get_ext_fullpath(dep)
                       for dep in depends]

        # Set proper filename generator
        get_ext_filename0 = self.get_ext_filename
        if isinstance(ext, SharedLibrary):
            self.get_ext_filename = self.get_ext_filename_shared_lib

        # OSX: change -bundle to -dynamiclib
        linker_so = []
        for arg in self.compiler.linker_so:
            linker_so += shlex.split(arg)
        linker_so = ['-dynamiclib' if arg == '-bundle' else arg
                     for arg in linker_so]
        self.compiler.set_executables(linker_so=linker_so)

        try:
            # Build extension
            super().build_extension(ext)
        finally:
            # Reset
            self.compiler = compiler0
            self.get_ext_filename = get_ext_filename0
            self.build_temp = build_temp0
            ext.depends = depends


class NVCCompiler(unixccompiler.UnixCCompiler):
    # Define a cuda compiler
    # Main difference is it uses `nvcc` instead of `$CC` and
    # knows about .cu files.
    compiler_type = 'nvcc'

    def __init__(self, verbose=0, dry_run=0, force=0):

        unixccompiler.UnixCCompiler.__init__(self, verbose, dry_run, force)

        home = cuda_home()
        nvcc = os.path.join(home, 'bin', 'nvcc')

        self.set_executables(compiler=nvcc,
                             compiler_so=nvcc,
                             compiler_cxx=nvcc,
                             linker_exe=nvcc,
                             linker_so='{} -shared'.format(nvcc))
        self.src_extensions += ['.cu']


# Below stuff is because we need to use our own `make_compiler` function
# that knows about nvcc. I didn't find a nice way to register the
# new NVCCompiler into the existing distutils stuff.

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
    fix_compile_parallel(compiler_object)

    return compiler_object
