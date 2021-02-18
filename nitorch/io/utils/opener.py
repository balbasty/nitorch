"""File opener for maybe compressed files.

Inspired by nibabel's openers (with some copied parts):
https://github.com/nipy/nibabel/blob/master/nibabel/openers.py
"""

from bz2 import BZ2File
from gzip import GzipFile, READ as gzip_r, WRITE as gzip_w
from distutils.version import StrictVersion
from warnings import warn
from os.path import splitext
import io

# is indexed_gzip present and modern?
IndexedGzipFile = None
try:
    import indexed_gzip as igzip
    version = igzip.__version__

    # < 0.7 - no good
    if StrictVersion(version) < StrictVersion('0.7.0'):
        warn(f'indexed_gzip is present, but too old (>= 0.7.0 required): {version})')
    # >= 0.8 SafeIndexedGzipFile renamed to IndexedGzipFile
    elif StrictVersion(version) < StrictVersion('0.8.0'):
        IndexedGzipFile = igzip.SafeIndexedGzipFile
    else:
        IndexedGzipFile = igzip.IndexedGzipFile
    del igzip, version
except ImportError:
    pass


def _gzip_open(filename, mode='rt', compresslevel=9, keep_open=False):
    if IndexedGzipFile and mode == 'rb':
        gzip_file = IndexedGzipFile(filename=filename, drop_handles=not keep_open)
    else:
        gzip_file = GzipFile(filename, mode, compresslevel)
    return gzip_file


def _mode(fileobj):
    if isinstance(fileobj, GzipFile):
        return 'rb' if fileobj.mode == gzip_r else 'wb'
    else:
        return fileobj.mode


def _is_fileobj(obj):
    """ Is `obj` a file-like object?"""
    return hasattr(obj, 'read') and hasattr(obj, 'write')


gz = dict(
    name='GZip',
    open=_gzip_open,
    argnames=('mode', 'compresslevel', 'keep_open'),
    defaults=('r', 1, False),
    accepted=(('rb', 'wb', 'xb', 'ab'), tuple(range(1, 10)), (True, False))
)

bz2 = dict(
    name='BZ2',
    open=BZ2File,
    argnames=('mode', 'compresslevel'),
    defaults=('rb', 1),
    accepted=(('rb', 'wb', 'xb', 'ab'), tuple(range(1, 10)))
)

base = dict(
    name='Base',
    open=open,
    argnames=('mode', 'buffering'),
    defaults=('r', -1),
    accepted=(('rt', 'r+t', 'wt', 'w+t', 'xt', 'at', 'a+t',
               'rb', 'r+b', 'wb', 'w+b', 'xb', 'ab', 'a+b'), tuple())
)


class Opener:
    """Generic opener

    - If given an opened file
      -> just manage it
    - If given a filename and used as a context manager
      -> if compressed, use appropriate opener
      -> open it and close it at the end
    - If given a filename and use as an `open` function
      -> if compressed, use appropriate opener
      -> open it
      -> close on deletion

    Some differences with the native `open`:
    - Default is binary mode, not tet mode.
      The user is expected to deal with different types of line endings,
      unless 't' mode is explicitly set,

    """

    # mapping from extension to opener
    ext_map = {'.gz': gz, '.bz2': bz2, None: base}

    def __init__(self, file_like, mode='r', opener=None, **kwargs):
        """

        Parameters
        ----------
        file_like : str of path-like or file-like
        mode : {'r', 'w', 'a', 'x', 'r+', 'w+', 'a+'}, default='r'
            Opening mode:
            - 'r' : open for reading, cursor at the beginning of the file
            - 'w' : open for writing, truncating the file first
            - 'a' : open for writing, cursor at the end of the file (no truncation)
            - 'x' : open for exclusive creation, failing if the file already exists
            The mode can be appended with '+', which opens the file for
            reading _and_ writing, while keeping the original behaviour:
            - 'r+' : open for reading and writing, cursor at the beginning of the file
            - 'w+' : open for reading and writing, truncating the file first
            - 'a+' : open for reading and writing, cursor at the end of the file (no truncation)
            The mode can furthermore be appended with:
            - 'b' : binary mode (default)
            - 't' : text mode
        opener : {gz, bz, base}, optional
            Specify which opener to use.
            If not provided, the file extension is used to decide.
        **kwargs : dict
            Additional opener-specific arguments,
        """
        self._name = None
        self.fileobj = None
        self.open = None

        if isinstance(file_like, Opener):
            if file_like.closed:
                if file_like.open:
                    self.fileobj = file_like.open()
                    self._name = file_like._name
                    self.open = file_like.open
                    return
                elif file_like.name:
                    self.__init__(file_like.name, mode)
                    return
                else:
                    raise ValueError('Cannot open a closed file descriptor')

        if _is_fileobj(file_like):
            if file_like.closed:
                if file_like.name:
                    self.__init__(file_like.name, mode)
                    return
                else:
                    raise ValueError('Cannot open a closed file descriptor')
            self.fileobj = file_like
            self._name = None
            self.open = lambda: self.fileobj
            return

        if not opener:
            ext = splitext(file_like)[-1].lower()
            if ext in self.ext_map.keys():
                opener = self.ext_map[ext]
            else:
                opener = self.ext_map[None]

        # set/pop options
        if not mode.endswith(('t', 'b')):
            mode = mode + 'b'
        for arg, allval, default in zip(opener['argnames'],
                                        opener['accepted'],
                                        opener['defaults']):
            if arg == 'mode':
                if allval and mode not in allval:
                    raise ValueError(f'Mode {mode} not possible with '
                                     f'opener {opener["name"]}.')
                kwargs['mode'] = mode
            else:
                kwargs.setdefault(default)
                if arg in kwargs:
                    if allval and kwargs[arg] not in allval:
                        raise ValueError(f'{arg} {kwargs[arg]} not possible '
                                         f'with opener {opener["name"]}.')
        for arg in list(kwargs.keys()):
            if arg not in opener['argnames']:
                kwargs.pop(arg)

        # open
        self._name = file_like
        self.fileobj = opener['open'](file_like, **kwargs)
        self.open = lambda: opener['open'](self._name, **kwargs)

    def __str__(self):
        return self.name or self.fileobj

    def __repr__(self):
        return f"Opener('{self.name or self.fileobj}', '{self.mode}')"

    @property
    def name(self):
        try:
            return self.fileobj.name
        except AttributeError:
            return self._name

    # Defer properties and methods
    is_indexed = property(lambda self: IndexedGzipFile and
                                       isinstance(self.fileobj, IndexedGzipFile))
    is_owned = property(lambda self: self._name is not None)
    closed = property(lambda self: self.fileobj.closed)
    mode = property(lambda self: _mode(self.fileobj))
    readable = lambda self: self.fileobj.readable()
    writable = lambda self: self.fileobj.writable()
    seekable = lambda self: self.fileobj.seekable()
    fileno = lambda self: self.fileobj.fileno()
    peek = lambda self, *a, **k: self.fileobj.peek(*a, **k)
    read = lambda self, *a, **k: self.fileobj.read(*a, **k)
    read1 = lambda self, *a, **k: self.fileobj.read1(*a, **k)
    readline = lambda self, *a, **k: self.fileobj.readline(*a, **k)
    readinto = lambda self, *a, **k: self.fileobj.readinto(*a, **k)
    readinto1 = lambda self, *a, **k: self.fileobj.readinto1(*a, **k)
    readlines = lambda self, *a, **k: self.fileobj.readlines(*a, **k)
    writelines = lambda self, *a, **k: self.fileobj.writelines(*a, **k)
    write = lambda self, *a, **k: self.fileobj.write(*a, **k)
    flush = lambda self, *a, **k: self.fileobj.flush(*a, **k)
    seek = lambda self, *a, **k: self.fileobj.seek(*a, **k)
    tell = lambda self, *a, **k: self.fileobj.tell(*a, **k)
    truncate = lambda self, *a, **k: self.fileobj.truncate(*a, **k) # NOT IN GZIP
    isatty = lambda self, *a, **k: self.fileobj.isatty(*a, **k)
    close = lambda self, *a, **k: self.fileobj.close(*a, **k) if self.fileobj else None
    close_if_mine = lambda self: self.close() if self.is_owned else None
    __iter__ = lambda self: iter(self.fileobj)
    __enter__ = lambda self: self
    __exit__ = lambda self, *a, **k: self.close_if_mine()
    __del__ = lambda self: self.close_if_mine()


def open(file_like, mode='r', opener=None, **kwargs):
    """Open a (potentially compressed) file.

    Parameters
    ----------
    file_like : str of path-like or file-like
    mode : {'r', 'w', 'a', 'x', 'r+', 'w+', 'a+'}, default='r'
        Opening mode:
        - 'r' : open for reading, cursor at the beginning of the file
        - 'w' : open for writing, truncating the file first
        - 'a' : open for writing, cursor at the end of the file (no truncation)
        - 'x' : open for exclusive creation, failing if the file already exists
        The mode can be appended with '+', which opens the file for
        reading _and_ writing, while keeping the original behaviour:
        - 'r+' : open for reading and writing, cursor at the beginning of the file
        - 'w+' : open for reading and writing, truncating the file first
        - 'a+' : open for reading and writing, cursor at the end of the file (no truncation)
        The mode can furthermore be appended with:
        - 't' : text mode (default)
        - 'b' : binary mode
    opener : {gz, bz, base}, optional
        Specify which opener to use.
        If not provided, the file extension is used to decide.
    **kwargs
        Additional opener-specific arguments,

    Returns
    -------
    Opener
        File-like object

    """
    return Opener(file_like, mode=mode, opener=opener, **kwargs)


class TransformedOpener(Opener):
    """Hack an existing file object so that it mimics an other mode."""

    def __init__(self, opener, mode):
        """

        Parameters
        ----------
        opener : Opener
        mode : str
        """
        self._name = None
        self.fileobj = None

        if not _is_fileobj(opener):
            raise TypeError('Expected a file-like object')

        if not mode.endswith(('t', 'b')):
            mode = mode + 't'

        compatible = True
        if mode[-1] not in opener.mode:
            compatible = False
        if '+' in mode:
            if not '+' in opener.mode:
                compatible = False
        if 'r' in mode:
            if not opener.readable():
                compatible = False
        if 'w' in mode:
            if not opener.writable:
                compatible = False
        if 'a' in opener.mode and 'a' not in mode:
            compatible = False

        if not compatible:
            raise ValueError(f'Cannot transform mode {opener.mode} into {mode}')

        self.fileobj = opener
        self._name = None
        self.open = lambda self: self.opener
        self._mode = mode

        if 'w' in self.mode:
            self.fileobj.seek(0)     # move to beginning of the file
            self.fileobj.truncate()
        elif 'a' in self.mode:
            self.fileobj.seek(0, 2)  # move to end of the file

    def __str__(self):
        return self.name or self.fileobj

    def __repr__(self):
        if not self.fileobj:
            return "TransformedOpener(None)"
        return f"TransformedOpener('{self.name or self.fileobj}', " \
               f"'{self.fileobj.mode}' -> '{self.mode}')"

    mode = property(lambda self: self._mode)
    readable = lambda self: 'r' in self.mode or '+' in self.mode
    writable = lambda self: 'w' in self.mode or 'a' in self.mode or '+' in self.mode
    seekable = lambda self: True

    def read(self, *args, **kwargs):
        if 'r' not in self.mode and '+' not in self.mode:
            return io.UnsupportedOperation('read')
        return self.fileobj.read(*args, **kwargs)

    def read1(self, *args, **kwargs):
        if 'r' not in self.mode and '+' not in self.mode:
            return io.UnsupportedOperation('read1')
        return self.fileobj.read1(*args, **kwargs)

    def readinto(self, *args, **kwargs):
        if 'r' not in self.mode and '+' not in self.mode:
            return io.UnsupportedOperation('readinto')
        return self.fileobj.readinto(*args, **kwargs)

    def readinto1(self, *args, **kwargs):
        if 'r' not in self.mode and '+' not in self.mode:
            return io.UnsupportedOperation('readinto1')
        return self.fileobj.readinto1(*args, **kwargs)

    def readline(self, *args, **kwargs):
        if 'r' not in self.mode and '+' not in self.mode:
            return io.UnsupportedOperation('readline')
        return self.fileobj.readline(*args, **kwargs)

    def readlines(self, *args, **kwargs):
        if 'r' not in self.mode and '+' not in self.mode:
            return io.UnsupportedOperation('readlines')
        return self.fileobj.readlines(*args, **kwargs)

    def write(self, buffer):
        if 'w' not in self.mode and 'a' not in self.mode  and '+' not in self.mode:
            return io.UnsupportedOperation('write')
        if 'a' in self.mode:
            where = self.fileobj.tell()
            self.fileobj.seek(0, 2)
            self.fileobj.write(buffer)
            self.fileobj.seek(where + len(buffer))
        else:
            return self.fileobj.write(buffer)

    def writelines(self, lines):
        if 'w' not in self.mode and 'a' not in self.mode  and '+' not in self.mode:
            return io.UnsupportedOperation('writelines')
        if 'a' in self.mode:
            where = self.fileobj.tell()
            self.fileobj.seek(0, 2)
            old_end = self.fileobj.tell()
            self.fileobj.writelines(lines)
            new_end = self.fileobj.tell()
            self.fileobj.seek(where + new_end - old_end)
        else:
            return self.fileobj.writelines(lines)


def transform_opener(opener, mode):
    """Transform an already opened file object so that it mimics another mode.

    Be careful, some conversions are not possible.

    Parameters
    ----------
    opener : Opener
    mode : str

    Returns
    -------
    TransformedOpener

    """
    return TransformedOpener(opener, mode)
