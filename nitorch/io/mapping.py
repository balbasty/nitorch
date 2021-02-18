from enum import Enum


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


class MappedFile:
    """Empty super class for all mapping classes.

    This is just so that the loaders know not to do anything if
    they are given a mapped object.
    """

    filename: str or dict = None      # Name of the file(s)
    fileobj: object or dict = None    # Object(s) with open/read/close
    shape: tuple = None               # Shape of the object returned by (f)data

    mode: str = 'r'                       # 'r' / 'r+'
    keep_open: bool = False               # keep file descriptor open?
    readable: AccessType = AccessType.No  # No/Full/Partial/TruePartial
    writable: AccessType = AccessType.No  # No/Full/Partial/TruePartial
    FailedReadError = RuntimeError        # class of errors raised on read
    FailedWriteError = RuntimeError       # class of errors raised no write

    def __init__(self, file_like, mode='r+', keep_open=False):
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
        pass

    def __str__(self):
        return '{}()'.format(type(self).__name__)

    __repr__ = __str__

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

