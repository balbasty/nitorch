from ..mapping import MappedNonLin
from nitorch.io.mappedfile import AccessType
from nitorch.io.utils.opener import open, Opener, transform_opener, gz
import numpy as np
from threading import RLock
from contextlib import contextmanager
from pathlib import Path
import torch


hdr_type = np.dtype([
    ('version', '<f4'),
    ('width', '<i4'),
    ('height', '<i4'),
    ('depth', '<i4'),
    ('spacing', '<i4'),
    ('exp_k', '<f4'),
])


class M3Z(MappedNonLin):

    readable: AccessType = AccessType.Full
    writable: AccessType = AccessType.Full
    Opener.ext_map['.m3z'] = gz
    dtype: torch.dtype = torch.float32

    @classmethod
    def possible_extensions(cls):
        return ('.m3z',)

    def __init__(self, file_like, mode='r', keep_open=True):
        """

        Parameters
        ----------
        file_like : str or fileobj
            Input file.
        mode : {'r', 'w', 'r+'}, default='r'
            File permission
        keep_open : bool, default=True
            Keep file descriptor open.
        """
        if mode not in ('r', 'w', 'r+', 'rb', 'wb', 'rb+'):
            raise ValueError(f"Mode expected in ('r', 'w', 'r+'). Got {mode}.")

        self.mode = mode            # Decides if the user lets us write
        self.keep_open = keep_open  # Keep file descriptor open (user value)?
        self._file_like = file_like
        self._fname = getattr(file_like, 'name', file_like)
        self._prepare_openers()

        with self.fileobj(self.mode) as f:
            self._hdr = np.fromfile(f, dtype=hdr_type, count=1)
        self._shape = (self._hdr['width'].item(),
                       self._hdr['height'].item(),
                       self._hdr['depth'].item(), 3)
        self._spacing = self._hdr['spacing'].item()
        self._exp_k = self._hdr['spacing'].item()

        super().__init__()

    def raw_data(self):


    # ------------------------------------------------------------------
    #    ADAPTED FROM NIBABEL'S ARRAYPROXY
    # ------------------------------------------------------------------

    @property
    def fname(self):
        return self._fname

    def _prepare_openers(self):
        self._opener = dict()       # Saved openers
        self._lock = None           # Lock (still unsure why I need that)

        mode = self.mode + 'b'
        self._lock = RLock()

        if hasattr(self._file_like, 'read') and hasattr(self._file_like, 'seek'):
            # file object -> keep stuff irrelevant
            if not self._file_like.readable() or not self._file_like.seekable():
                raise ValueError('File must be readable and seekable')
            if '+' in self.mode and not self._file_like.writable():
                raise ValueError('File must be writable in mode "r+"')
            self._opener[self.mode] = self._file_like
            return

        try:
            if self.keep_open:
                try:
                    self._opener[mode] = open(self._file_like, mode, keep_open=True)
                except ValueError:
                    self._opener[mode] = open(self._file_like, 'rb', keep_open=True)
            else:
                self._opener['r'] = open(self._file_like, 'rb', keep_open=False)
                if not self._opener['r'].is_indexed:
                    del self._opener['r']
        except FileNotFoundError:
            return

    @contextmanager
    def fileobj(self, mode='', seek=None):
        """Return an `Opener`.

        It can be used with:
        >> nii = BabelArray('/path/to/file')
        >> with nii.fileobj('image') as f:
        >>    f.seek(pos)
        >>    ...

        Parameters
        ----------
        mode : {'r', 'r+', 'w', 'w+', 'a', 'a+'}, default='r'
            Opening mode. The file type ('b' or 't') should be omitted.
        seek : int, optional
            Position to seek.

        """
        mode = mode or 'r'
        opener = None
        # check if we have the right opener
        if mode in self._opener:
            opener = self._opener[mode]
        # otherwise, check if we have one with more permissions than needed
        if not opener and mode + '+' in self._opener:
            opener = self._opener[mode + '+']
        # otherwise, check if we can hack a liberal one
        # (e.g., 'r+' -> 'w(+)' or 'a(+)')
        if not opener:
            for mode0, opener0 in self._opener.items():
                try:
                    opener = transform_opener(opener0, mode)
                    break
                except ValueError:
                    pass
        # we found one -> perform a few sanity checks
        if opener:
            check = True
            if 'r' in mode or '+' in mode:
                check = check and opener.readable()
            if 'w' in mode or 'a' in mode:
                check = check and opener.writable()
            if check:
                if seek is not None:
                    opener.seek(seek)
                yield opener
                return
        # everything failed -> create one from scratch
        with open(self._file_like, mode=mode, keep_open=False) as opener:
            check = True
            if 'r' in mode or '+' in mode:
                check = check and opener.readable()
            if 'w' in mode or 'a' in mode:
                check = check and opener.writable()
            if check:
                if seek is not None:
                    opener.seek(seek)
                yield opener
                return

        raise RuntimeError('Could not yield an appropriate file object')

    def close(self):
        if hasattr(self, '_opener'):
            for key, opener in self._opener.items():
                if not opener.closed:
                    opener.close()
        super().close()
        return self

    def close_if_mine(self):
        if hasattr(self, '_opener'):
            for key, opener in self._opener.items():
                if not opener.closed:
                    opener.close_if_mine()
        super().close_if_mine()
        return self

    def __del__(self):
        """If this ``ArrayProxy`` was created with ``keep_file_open=True``,
        the open file object is closed if necessary.
        """
        self.close_if_mine()
        if hasattr(self, '_opener'):
            del self._opener

    def __getstate__(self):
        """Returns the state of this ``ArrayProxy`` during pickling. """
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        """Sets the state of this ``ArrayProxy`` during unpickling. """
        self.__dict__.update(state)
        self._lock = RLock()
