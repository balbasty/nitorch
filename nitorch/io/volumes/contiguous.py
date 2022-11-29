from .mappedarray import MappedArray
from ..mappedfile import AccessType, FileInfo
from ..utils.opener import open, Opener, transform_opener, is_compressed_fileobj
from ..utils.indexing import is_fullslice, splitop
from ..utils.volumeio import read_array, write_array, read_subarray, write_subarray
from ..utils import volumeutils as volutils
from ..loadsave import map as map_array
from nitorch.core import py, dtypes
import numpy as np
from threading import RLock
from contextlib import contextmanager
from pathlib import Path
import torch
import sys


class ContiguousArray(MappedArray):
    """Base class for contiguous array readers"""

    _hdr: np.ndarray = None             # Header
    _ftr: np.ndarray = None             # Footer
    _offset_hdr: int = 0                # Offset to beginning of header
    _offset_dat: int = 0                # Offset to beginning of data
    _offset_ftr: int = 0                # Offset to beginning of footer
    _dtype_hdr: np.dtype = None         # Header (structured) datatype
    _dtype_dat: np.dtype = None         # Data (element) datatype
    _dtype_ftr: np.dtype = None         # Header (structured) datatype
    _order: str = 'C'                   # Data order on disk

    def __init__(self, file_like, mode='r', keep_open=False, **kwargs):
        self.filemap['data'] = FileInfo(
            openers={},
            filelike=file_like,
            mode=mode,
            keep_open=keep_open,
            readable=AccessType.Full,
            writable=AccessType.No,
        )
        self._prepare_openers(data=file_like)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    #    HIGH-LEVEL IMPLEMENTATION
    # ------------------------------------------------------------------
    # These functions implement the MappedArray API
    # (that's why there's no doc, it's at the MappedArray level)

    def set_data(self, dat, casting='unsafe'):
        if '+' not in self.mode:
            raise RuntimeError('Cannot write into read-only volume. '
                               'Re-map in mode "r+" to allow in-place '
                               'writing.')

        # --- convert to numpy ---
        if torch.is_tensor(dat):
            dat = dat.detach().cpu()
        dat = np.asanyarray(dat)

        # --- sanity check ---
        if dat.shape != self.shape:
            raise ValueError('Expected an array of shape {} but got {}.'
                             .format(self.shape, dat.shape))

        # --- special case if empty view ---
        if dat.size == 0:
            # nothing to write
            return self

        # --- cast ---
        dat = volutils.cast(dat, self.dtype, casting)

        # --- unpermute ---
        drop, perm, slicer = splitop(self.permutation, self.index, 'w')
        dat = dat[drop].transpose(perm)

        # --- dispatch ---
        if self.filemap['data'].is_compressed:
            if not is_fullslice(slicer, self._shape):
                # read-and-write
                slice = dat
                dat = self._read_data_raw()
                dat[slicer] = slice
            with self.filemap['data'].fileobj('w', seek=0) as f:
                if self.same_file('data', 'header'):
                    self._write_header_raw(fileobj=f)
                self._write_data_raw_full(dat, fileobj=f)
                if self.same_file('data', 'footer'):
                    self._write_footer_raw(fileobj=f)

        elif is_fullslice(slicer, self.shaped.shape):
            with self.filemap['data'].fileobj('r+') as f:
                self._write_data_raw_full(dat, fileobj=f)

        else:
            with self.fileobj('image', 'r+') as f:
                self._write_data_raw_partial(dat, slicer, fileobj=f)

        return self

    def raw_data(self):
        # --- check that view is not empty ---
        if py.prod(self.shape) == 0:
            return np.zeros(self.shape, dtype=self.dtype)

        # --- read native data ---
        slicer, perm, newdim = splitop(self.permutation, self.index, 'r')
        with self.filemap['data'].fileobj('r') as f:
            dat = self._read_data_raw(slicer, fileobj=f)
        dat = dat.transpose(perm)[newdim]
        return dat

    def metadata(self, keys=None):
        return self.from_header(keys)

    def set_metadata(self, **meta):
        if '+' not in self.mode:
            raise RuntimeError('Cannot write into read-only volume. '
                               'Re-map in mode "r+" to allow in-place '
                               'writing.')

        self.update_header(**meta)

        if self.filemap['header'].is_compressed:
            # read-and-write
            if self.same_file('data', 'header'):
                with self.filemap['header'].fileobj('r') as f:
                    full_dat = self._read_data_raw(fileobj=f)
            with self.filemap['header'].fileobj('w') as f:
                self._write_header_raw(fileobj=f)
                if self.same_file('header', 'data'):
                    self._write_data_raw_full(full_dat, fileobj=f)
                if self.same_file('header', 'footer'):
                    self._write_footer_raw(fileobj=f)
        else:
            with self.filemap['header'].fileobj('r+') as f:
                self._write_header_raw(fileobj=f)

    # ------------------------------------------------------------------
    #    LOW-LEVEL IMPLEMENTATION
    # ------------------------------------------------------------------
    # the following functions implement read/write of (meta)data at the
    # lowest level (no conversion is performed there)

    def _write_data_raw_partial(self, dat, slicer, fileobj=None, key=None, lock=None):
        key = key or next(iter(self.filemap))
        lock = lock or self.filemap[key].lock

        if fileobj is None:
            # Create our own context
            with self.filemap[key].fileobj('r+') as fileobj:
                return self._write_data_raw_partial(dat, slicer, fileobj, key, lock)

        # sanity checks for developers
        # (asserts because proper conversion/dispatch should happen before)
        assert isinstance(dat, np.ndarray), "Data should already be numpy"
        assert dat.dtype == self.dtype, "Data should already have correct type"
        assert not self.filemap[key].is_compressed, "Data cannot be compressed"
        assert not is_fullslice(slicer, self.shaped.shape), "No need for partial writing"

        if not fileobj.readable():
            raise RuntimeError('File object not readable')
        if not fileobj.writable():
            raise RuntimeError('File object not writable')

        # write data chunk
        write_subarray(dat, fileobj, tuple(slicer), self.shaped.shape, self.dtype,
                       offset=self._offset_dat,
                       order=self._order,
                       lock=lock)

    def _write_data_raw_full(self, dat, fileobj=None, key=None, lock=None):
        key = key or next(iter(self.filemap))
        lock = lock or self.filemap[key].lock

        if fileobj is None:
            # Create our own context
            with self.fileobj(key, 'a') as fileobj:
                return self._write_data_raw_full(dat, fileobj, key, lock)

        # sanity checks for developers
        # (asserts because proper conversion/dispatch should happen before)
        assert isinstance(dat, np.ndarray), "Data should already be numpy"
        assert dat.dtype == self.dtype, "Data should already have correct type"

        if not fileobj.writable():
            raise RuntimeError('File object not writable')

        write_array(dat, fileobj, self.dtype,
                    offset=self._offset_dat, order=self._order)

    def _read_data_raw_full(self, fileobj=None, lock=None, key=None):
        key = key or next(iter(self.filemap))
        lock = lock or self.filemap[key].lock

        if fileobj is None:
            with self.fileobj(key, 'r') as fileobj:
                return self._read_data_raw_full(fileobj, lock, key)

        return read_array(fileobj, self.shaped.shape, self.dtype, fileobj,
                          offset=self._offset_dat, order=self._order)

    def _read_data_raw_partial(self, slicer, fileobj=None, lock=None, key=None):
        key = key or next(iter(self.filemap))
        lock = lock or self.filemap[key].lock

        if fileobj is None:
            with self.fileobj(key, 'r') as fileobj:
                return self._read_data_raw_partial(slicer, fileobj, lock, key)

        return read_subarray(fileobj, slicer, self._shape, self.dtype,
                             offset=self._offset_dat,
                             order=self._order,
                             lock=lock)

    def _read_data_raw(self, slicer=None, fileobj=None):
        if not slicer or is_fullslice(slicer, self.shaped.shape):
            dat = self._read_data_raw_full(fileobj)
            if slicer:
                dat = dat[slicer]
        else:
            dat = self._read_data_raw_partial(slicer, fileobj)
        return dat

    # ------------------------------------------------------------------
    #    ADAPTED FROM NIBABEL'S ARRAYPROXY
    # ------------------------------------------------------------------
    # I am bypassing ArrayProxy to read and -- mainly -- write
    # data from/to disk. I've therefore copied the methods that deal
    # with file opening/permissions/locks.
    # Here, we want to write partially into existing files, so keeping
    # files open (including header files) is more common than in
    # vanilla nibabel.

    def same_file(self, *keys):
        filenames = [self.filemap[key].filename for key in keys]
        return len(set(filenames)) == 1

    def _prepare_openers(self, **filelike):
        for key, val in filelike.items():
            self.filemap.setdefault(key, FileInfo())
            fileinfo = self.filemap[key]
            mode = self.filemap[key].mode
            file_like = val.file_like
            fileinfo.lock = RLock()
            if hasattr(file_like, 'read') and hasattr(file_like, 'seek'):
                # file object -> keep stuff irrelevant
                if not file_like.readable() or not file_like.seekable():
                    raise ValueError('File must be readable and seekable')
                if '+' in mode and not file_like.writable():
                    raise ValueError('File must be writable in mode "r+"')
                fileinfo.openers[mode] = file_like
                continue
            try:
                if fileinfo.keep_open:
                    try:
                        fileinfo.openers[mode] \
                            = open(file_like, mode + 'b', keep_open=True)
                    except ValueError:
                        fileinfo.openers[mode] \
                            = open(file_like, 'rb', keep_open=True)
                else:
                    fileinfo.openers['r'] \
                        = open(file_like, 'rb', keep_open=False)
                    if not fileinfo.openers['r'].is_indexed:
                        del fileinfo.openers['r']
            except FileNotFoundError:
                continue
