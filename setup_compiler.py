"""Get info on available compilers"""
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
from distutils.unixccompiler import UnixCCompiler
import os
import subprocess
import re
import setup_utils


def guess_compiler(platform=None, compiler=None):
    """Guess the (customized) C compiler object that will be used by
    distutils to compile our extensions.

    Parameters
    ----------
    platform : {'posix', 'nt'}, default=`os.name`
        Underlying platform.
    compiler : {'unix', 'msvc', 'cygwin', 'mingw32', 'bcpp'}, optional
        A string that describes one of the `CCompiler` classes.
        If not provided, it is selected based on the platform:
            - 'posix' -> 'unix'
            - 'nt' -> 'msvc'

    Returns
    -------
    ccompiler : CCompiler

    """
    ccompiler = new_compiler(platform, compiler)
    customize_compiler(ccompiler)
    return ccompiler


def guess_compiler_name(compiler=None):
    """Guess the name of a compiler.

    A single `CCompiler` object can be using multiple compilers
    under the hood (typically, the `UnixCCompiler` calls the compiler
    defined in `$CC`, which can be gcc, clang or something else). This
    functions calls the underlying compiler (after cutomization of
    the `CCompiler` class) and parses its output to guess its name.

    If parsing it fails, we just return the name of the binary.

    Parameters
    ----------
    compiler : distutils.CCompiler, optional
        Compiler object as returned by `guess_compiler`

    Returns
    -------
    {'gcc', 'clang', 'clang-apple', 'msvc', 'bcpp', ...}
    """
    if compiler is None:
        compiler = guess_compiler()
    if isinstance(compiler, UnixCCompiler):
        exe = compiler.compiler
        if not isinstance(exe, str):
            exe = exe[0]
        # gcc can be clang on macos -> better to run --version to check
        try:
            with open(os.devnull, 'w') as devnull:
                version = subprocess.check_output([exe, '--version'],
                                                  stderr=devnull)
            version = version.decode()
            for line in version.split('\n'):
                # do we recognize gcc?
                match = re.search('\(.*GCC.*\)', line)
                if match:
                    return 'gcc'
                # do we recognize apple clang?
                match = re.search('Apple clang', line)
                if match:
                    return 'clang-apple'
                # do we recognize normal clang?
                match = re.search('clang', line)
                if match:
                    return 'clang'
        except Exception:
            pass
        return os.path.basename(exe)
    elif type(compiler) == 'BCPPCompiler':
        return 'bcpp'
    elif type(compiler) == 'MSVCCompiler':
        return 'msvc'


def guess_compiler_version(compiler=None):
    """Guess the version of a compiler.

    Parameters
    ----------
    compiler : distutils.CCompiler, optional
        Compiler object as returned by `guess_compiler`

    Returns
    -------
    setup_utils.version
        Version object

    """

    notfound = setup_utils.version()

    def read_version(exe):
        with open(os.devnull, 'w') as devnull:
            version = subprocess.check_output(exe, stderr=devnull)
            version = version.decode()
            return setup_utils.version.from_str(version)

    def parse_version(exe, pattern):
        try:
            print(exe)
            with open(os.devnull, 'w') as devnull:
                version = subprocess.check_output(exe, stderr=devnull)
            version = version.decode()
            for line in version.split('\n'):
                match = re.search(pattern, line)
                if match:
                    version = match.group('version')
                    return setup_utils.version.from_str(version)
        except Exception:
            pass
        return notfound

    if compiler is None:
        compiler = guess_compiler()
    name = guess_compiler_name(compiler)

    if name.startswith('gcc') or name.startswith('clang'):
        exe = compiler.compiler
        if not isinstance(exe, str):
            exe = exe[0]
        return read_version([exe, '-dumpversion'])

    elif name == 'msvc':
        exe = compiler.cc
        if not isinstance(exe, str):
            exe = exe[0]
        pattern = r'Version (?P<version>[0-9]+(\.[0-9]+)*)'
        return parse_version([exe, '-v'], pattern)

    elif name == 'bcpp':
        exe = compiler.cc
        if not isinstance(exe, str):
            exe = exe[0]
        pattern = r'Borland C\+\+ (?P<version>[0-9]+(\.[0-9]+)*)'
        return parse_version(exe, pattern)

    else:
        return notfound
