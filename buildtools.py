"""distutils/setuptools extension."""

import os
import os.path
import sys
import distutils
import subprocess
import glob
import re
from packaging import version as pversion
from distutils import ccompiler, unixccompiler
from distutils.command import build_clib as build_clib_base

_all__ = [
    'is_windows',
    'is_darwin',
    'cuda_home',
    'cuda_version',
    'cudnn_home',
    'cudnn_version',
    'build_clib',
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
    version = (None, None, None)
    for line in lines:
        if version[0] is None:
            version[0] = search_define('CUDNN_MAJOR', line, version[0])
        if version[1] is None:
            version[1] = search_define('CUDNN_MINOR', line, version[1])
        if version[2] is None:
            version[2] = search_define('CUDNN_PATCHLEVEL', line, version[2])
    return version


class build_shared_clib(build_clib_base.build_clib):
    """Extends native lib builder to handle compiler flags."""

    def initialize_options(self):
        super().initialize_options()
        self.extra_compile_args = None
        self.extra_link_args = None
        self.extra_objects = None
        self.export_symbols = None
        self.runtime_library_dirs = None
        self.depends = None
        self.language = None
        self.package = None

    def run(self):
        if not self.libraries:
            return

        if self.package is None:
            self.package = self.distribution.ext_package


        self.build_libraries(self.libraries)


    def build_libraries(self, libraries):
        compiler0 = self.compiler
        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                from distutils.errors import DistutilsSetupError
                raise DistutilsSetupError(
                   "in 'libraries' option (library '%s'), "
                   "'sources' must be present and must be "
                   "a list of source filenames" % lib_name)
                sources = list(sources)

            distutils.log.info("building '%s' library", lib_name)

            compiler = build_info.get('compiler') or compiler0
            language = build_info.get('language') or self.language
            self.compiler = make_compiler(compiler=compiler,
                                          language=language,
                                          dry_run=self.dry_run,
                                          force=self.force)

            distutils.sysconfig.customize_compiler(self.compiler)

            if self.include_dirs is not None:
                self.compiler.set_include_dirs(self.include_dirs)
            if self.define is not None:
                # 'define' option is a list of (name,value) tuples
                for (name, value) in self.define:
                    self.compiler.define_macro(name, value)
            if self.undef is not None:
                for macro in self.undef:
                    self.compiler.undefine_macro(macro)

            # (YB: copied from build_ext)
            # Two possible sources for extra compiler arguments:
            #   - 'extra_compile_args' in Extension object
            #   - CFLAGS environment variable (not particularly
            #     elegant, but people seem to expect it and I
            #     guess it's useful)
            # The environment variable should take precedence, and
            # any sensible compiler will give precedence to later
            # command line args.  Hence we combine them in order:
            extra_args = build_info.get('extra_compile_args') or []

            # First, compile the source code to object files in the library
            # directory.  (This should probably change to putting object
            # files in a temporary build directory.)
            macros = build_info.get('macros')
            include_dirs = build_info.get('include_dirs')
            depends = build_info.get('depends')
            objects = self.compiler.compile(sources,
                                            output_dir=self.build_temp,
                                            macros=macros,
                                            include_dirs=include_dirs,
                                            debug=self.debug,
                                            extra_postargs=extra_args,
                                            depends=depends)

            # Now link the object files together into a "shared object" --
            # of course, first we have to figure out all the other things
            # that go into the mix.
            if build_info.get('depends'):
                objects.extend(build_info.get('depends'))
            extra_args = build_info.get('extra_linking_args') or []

            # Detect target language, if not provided
            language = build_info.get('language') \
                or self.compiler.detect_language(sources)

            runtime_library_dirs = build_info.get('runtime_library_dirs')
            library_dirs = build_info.get('library_dirs')
            export_symbols = build_info.get('export_symbols')
            libraries = build_info.get('libraries')

            lib_fullpath = self.compiler.library_filename(
                lib_name, output_dir=os.path.join(self.package, 'lib'),
                lib_type='shared')

            self.compiler.link_shared_object(
                objects, lib_fullpath,
                libraries=libraries,
                library_dirs=library_dirs,
                runtime_library_dirs=runtime_library_dirs,
                extra_postargs=extra_args,
                export_symbols=export_symbols,
                debug=self.debug,
                build_temp=self.build_temp,
                target_lang=language)

        # Restore original compiler
        self.compiler = compiler0


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