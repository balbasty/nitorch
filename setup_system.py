import os
import sys


def guess_platform():
    """{'win32', 'cygwin', 'darwin', 'linux', 'aix', ...}
    Platforms whose name is not in this list are all non-linux unix systems.
    """
    return sys.platform


def guess_os_type():
    """{'posix', 'nt', 'java'}
    This specifies which platform-specific library is available.
    - 'posix'   is available on posix-compliant systems (unix-like, cygwin).
    - 'nt'      is available on pure windows systems (win32)
    - 'java'    is available on java VMs (jython?)
    """
    return os.name


def is_win32(platform=None):
    """True if pure windows (win32 -> nt)."""
    platform = platform or guess_platform()
    return platform.startswith('win32')


def is_cygwin(platform=None):
    """True if cygwin windows (cygwin -> posix)."""
    platform = platform or guess_platform()
    return platform.startswith('cygwin')


def is_windows(platform=None):
    """True if windows system (win32 of cygwin)."""
    platform = platform or guess_platform()
    return is_win32(platform) or is_cygwin(platform)


def is_darwin(platform=None):
    """True if macos (darwin -> posix)"""
    platform = platform or sys.platform
    return platform.startswith('darwin')


def is_linux(platform=None):
    """True if linux (more restrictive than unix)"""
    platform = platform or sys.platform
    return not is_windows(platform) and not is_darwin(platform)


def is_unix(platform=None):
    """True if unix (unix, linux, darwin, aix, ...)"""
    platform = platform or sys.platform
    return is_linux(platform) or is_darwin(platform)


def is_unix_like(platform=None):
    """True if unix-like (unix + cygwin)"""
    platform = platform or sys.platform
    return is_unix(platform) or is_cygwin(platform)
