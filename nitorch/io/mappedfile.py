from enum import Enum
from typing import Dict
from .utils.opener import Opener, OpenMode, transform_opener
from contextlib import contextmanager
from threading import RLock


class AccessType(int, Enum):
    """Enumeration that describes the type of access that is implemented

        * No : Read (or write) is not possible at all
        * Full : Only the full volume can be read (or written)
        * Partial : The volume can be read (or written) partially.
            However, it is possible than, under the hood, a full read
            (or write) is triggered.
        * TruePartial : The volume can be read (or written) partially.
            It is ensured that data is really accessed partially,
            in an efficient manner.

        These values map to {0, 1, 2, 3}. Therefore, they are *ordered*
        (`Partial > Full`) and only `No` evaluates to `False`
        (`bool(No) -> False`, `bool(Full) -> True`).
    """

    @staticmethod
    def _name2inst(val):
        if isinstance(val, str):
            val = val.lower()
            if val == 'no':
                val = 0
            elif val == 'full':
                val = 1
            elif val == 'partial':
                val = 2
            elif val == 'truepartial':
                val = 3
            else:
                raise ValueError('{} does not map to any value'.format(val))
        return AccessType(val)

    def __eq__(self, other):
        try:
            return self.value == self._name2inst(other).value
        except ValueError:
            return False

    def __ne__(self, other):
        try:
            return self.value != self._name2inst(other).value
        except ValueError:
            return True

    def __ge__(self, other):
        return self.value >= self._name2inst(other).value

    def __gt__(self, other):
        return self.value > self._name2inst(other).value

    def __le__(self, other):
        return self.value <= self._name2inst(other).value

    def __lt__(self, other):
        return self.value < self._name2inst(other).value

    No = 0
    Full = 1
    Partial = 2
    TruePartial = 3


class FileInfo:
    filename: str = None                  # filename (can be None if in-memory proxy)
    filelike = None                       # file-like objects (`write`, `seek`, etc)
    openers: Dict[str, Opener] = {}       # opener objects (`write`, `seek`, etc)
    lock = None                           # Lock for multithreaded access
    is_compressed: bool = None            # is compressed
    mode: str = 'r'                       # 'r' / 'r+'
    keep_open: bool = False               # keep file descriptor open?
    readable: AccessType = AccessType.No  # No/Full/Partial/TruePartial
    writable: AccessType = AccessType.No  # No/Full/Partial/TruePartial

    def __init__(self, **kwargs):
        keys = ('filename', 'filelike', 'openers', 'is_compressed', 'mode',
                'keep_open', 'readable', 'writeable')

        # set user-defined
        for key, value in kwargs.items():
            setattr(self, key, value)

        # set defaults
        for key in keys:
            if key in kwargs:
                continue
            setattr(self, key, getattr(type(self), key))

    @contextmanager
    def fileobj(self, mode=None, seek=None):
        mode = mode or self.mode
        if OpenMode(mode) > OpenMode(self.mode):
            raise ValueError(f'Cannot open in mode {mode}, which is '
                             f'less restrictive than mode {self.mode}')

        # check if we have the right opener
        opener = self.openers.get(mode, None)
        # otherwise, check if we have one with more permissions than needed
        opener = opener or self.openers.get(mode + '+')
        # otherwise, check if we can hack a liberal one
        # (e.g., 'r+' -> 'w(+)' or 'a(+)')
        if not opener:
            for mode0, opener0 in self.openers.items():
                try:
                    opener = transform_opener(opener0, mode)
                    break
                except ValueError:
                    pass

        def check_opener(opener):
            check = True
            if OpenMode.readable(mode):
                check = check and opener.readable()
            if OpenMode.writable(mode):
                check = check and opener.writable()
            return check

        # we found one -> perform a few sanity checks
        if opener and check_opener(opener):
            if seek is not None:
                opener.seek(seek)
            yield opener
            return

        # everything failed -> create one from scratch (from fileobj)
        filelike = self.filelike
        try:
            with Opener(filelike, mode=mode, keep_open=False) as opener:
                if check_opener(opener):
                    if seek is not None:
                        opener.seek(seek)
                    yield opener
                    return
        except Exception:
            pass

        # everything failed -> create one from scratch (from filename)
        filelike = self.filename
        try:
            with Opener(filelike, mode=mode, keep_open=False) as opener:
                if check_opener(opener):
                    if seek is not None:
                        opener.seek(seek)
                    yield opener
                    return
        except Exception:
            pass


def _defer2dict(key, default=None):
    def get(self):
        if not self.filemap:
            return default
        return getattr(next(iter(self.filemap.values())), key)
    def set(self, value):
        return setattr(next(iter(self.filemap.values())), key, value)
    return property(get, set)


class MappedFile:
    """Empty super class for all mapping classes.

    This is just so that the loaders know not to do anything if
    they are given a mapped object.
    """

    filemap: Dict[str, FileInfo] = {}
    filename = _defer2dict('filename')
    fileobj = _defer2dict('fileobj')
    mode = _defer2dict('mode', 'r')
    keep_open = _defer2dict('keep_open', False)
    readable = _defer2dict('readable', AccessType.No)
    writable = _defer2dict('writable', AccessType.No)

    shape: tuple = None               # Shape of the object returned by (f)data
    FailedReadError = RuntimeError    # class of errors raised on read
    FailedWriteError = RuntimeError   # class of errors raised no write

    def __init__(self, file_like=None, mode='r+', keep_open=False, **kwargs):
        """

        Parameters
        ----------
        file_like : str of file object
            File to map
        mode : {'r', 'r+'}, default='r'
            Read in read-only ('r') or read-and-write ('r+') mode.
            Modifying the file in-place is only possible in 'r+' mode.
            Note that 'r+' mode may not be implemented in all formats.
            If not implemented, `set_data` and `set_fdata` will fail.
        keep_open : bool, default=False
            Keep file open. Can be more efficient if multiple reads (or
            writes) are expected. Note that this option may not be
            implemented in all formats. If not implemented, does nothing.
        """
        if file_like is None:
            self.init(**kwargs)
        else:
            self.init(file_like=file_like, mode=mode, keep_open=keep_open,
                      **kwargs)

    def init(self, **kwargs):
        for key, val in kwargs:
            setattr(self, key, val)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_if_mine()

    def __del__(self):
        self.close_if_mine()

    def close(self):
        for k, v in self.filemap:
            for mode, opener in v.openers.items():
                if not opener.closed:
                    opener.close()
        return self

    def close_if_mine(self):
        for k, v in self.filemap:
            for mode, opener in v.openers.items():
                if not opener.closed:
                    opener.close_if_mine()
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        for k, v in state['filemap'].items():
            state['filemap'][k] = state['filemap'][k].__dict__.copy()
            state['filemap'][k].pop('lock')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k, v in self.filemap.items():
            self.filemap[k] = FileInfo()
            self.filemap[k].__dict__.update(v)
            self.filemap[k].lock = RLock()

    def __str__(self):
        return '{}()'.format(type(self).__name__)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def possible_extensions(cls):
        """List all possible extensions"""
        return tuple()

    @classmethod
    def sniff(self, file_like):
        """Quick-check that a file is of a given format.

        When this function returns True, it is not assured that the read
        will succeed. The point is rather to quickly detect when a file
        does *not match a format.

        By default, this function returns True, and read is tried.
        """
        return True

    def fdata(self, *args, **kwargs):
        """Load pre-processed data.

        Pre-processing is file-type specific (e.g., volume, transformation)
        and may involve casting to a specific data type, ensuring that
        some shared conventions are followed, etc.
        See implementation classes for specifics.
        """
        raise NotImplementedError

    def data(self, *args, **kwargs):
        """Load raw data.

        What is considered raw data is file-type specific (e.g., volume,
        transformation) and may involve keeping the original data type
        or returning minimally pre-processed data.
        See implementation classes for specifics.
        """
        raise NotImplementedError

