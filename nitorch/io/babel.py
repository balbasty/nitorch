"""Implementation of MappedArray based on nibabel.

..warning:: This file assumes that nibabel (and numpy) is available and
            should not be imported if it is not the case!
"""


# python imports
import os
import sys
from warnings import warn
from contextlib import contextmanager
from threading import RLock
# external imports
import torch
import numpy as np
import nibabel as nib
# nibabel imports
from nibabel import openers
from nibabel.spatialimages import SpatialImage
from nibabel.imageclasses import all_image_classes
from nibabel.volumeutils import _is_compressed_fobj as is_compressed_fobj
from nibabel.filebasedimages import ImageFileError
# nitorch imports
from nitorch.core import pyutils
# local imports
from .mapping import MappedArray
from .readers import reader_classes
from .indexing import invert_permutation, is_newaxis, is_sliceaxis, \
                      is_droppedaxis, is_broadcastaxis
from . import dtype as cast_dtype
from .metadata import keys as metadata_keys
from ._babel_metadata import header_to_metadata, metadata_to_header
from ._babel_utils import writeslice, is_full, array_to_file, \
                          full_heuristic, threshold_heuristic, _NullLock


class BabelArray(MappedArray):
    """MappedArray that relies on NiBabel."""

    def __init__(self, file_like, perm='r+', keep_open=True):
        """

        Parameters
        ----------
        file_like : str or fileobj
            Input file.
        perm : {'r', 'r+', 'w'}, default='r+'
            Permission
        keep_open : bool, default=True
            Keep file open.
        """

        if nib is None:
            raise ImportError('NiBabel is not available.')

        if isinstance(file_like, SpatialImage):
            self._image = file_like
        else:
            self._image = nib.load(file_like, mmap=False,
                                   keep_file_open=keep_open)

        # deal with file openers
        self._mode = perm
        self._keep_open = dict()
        self._keep_opener = dict()
        self._opener = dict()
        self._lock = dict()
        for key, val in self._image.file_map.items():
            self._keep_open[key], self._keep_opener[key] \
                = self._image.dataobj._should_keep_file_open(val.file_like, keep_open)
            self._lock[key] = RLock()

        super().__init__()

    @classmethod
    def possible_extensions(cls):
        ext = []
        for klass in all_image_classes:
            ext.append(*klass.valid_exts)
        return tuple(ext)

    FailedReadError = ImageFileError

    # ------------------------------------------------------------------
    #    ATTRIBUTES
    # ------------------------------------------------------------------
    # A bunch of attributes are known by nibabel. I just link to them
    # when that's the case

    fname = property(lambda self: self._image.file_map['image'].filename)
    _affine = property(lambda self: torch.as_tensor(self._image.affine))
    _spatial = property(lambda self: tuple([True]*3 + [False]*max(0, self._dim-3)))
    _shape = property(lambda self: self._image.shape)
    dtype = property(lambda self: self._image.dataobj.dtype)
    slope = property(lambda self: self._image.dataobj.slope)
    inter = property(lambda self: self._image.dataobj.inter)

    @slope.setter
    def slope(self, val):
        self._image.dataobj.slope = float(np.asarray(val).item())

    @inter.setter
    def inter(self, val):
        self._image.dataobj.inter = float(np.asarray(val).item())

    def is_compressed(self, key='image'):
        with self.fileobj(key) as f:
            if isinstance(f, nib.openers.Opener):
                f = f.fobj
            iscomp = is_compressed_fobj(f)
        return iscomp

    # ------------------------------------------------------------------
    #    LOW-LEVEL IMPLEMENTATION
    # ------------------------------------------------------------------
    # the following functions implement read/write of (meta)data at the
    # lowest level (no conversion is performed there)

    def _set_data_raw(self, dat):
        """Write native data"""

        # 0) convert to numpy
        if torch.is_tensor(dat):
            dat = dat.detach().cpu().numpy()
        dat = np.asanyarray(dat)

        # sanity checks
        slicer = self.slicer or [slice(None)]*self.dim
        slicer_nonew = [idx for idx in slicer if not is_newaxis(idx)]
        if any(is_broadcastaxis(idx) for idx in slicer):
            raise ValueError('Cannot write into a broadcasted volume.')
        if not(dat.dtype.kind == self.dtype.kind and
               dat.dtype.itemsize == self.dtype.itemsize):
            raise TypeError('Expected array with data type {} but got {}.'
                            .format(dat.dtype, self.dtype))
        if self.is_compressed('image') and not is_full(self.slicer, self._shape):
            # we read-and-write the full dataset
            heuristic = full_heuristic
        else:
            # we can read partially into the dataset
            heuristic = threshold_heuristic

        # 1) remove new axes
        dim_map_nodropped = [d for d, idx in zip(self.permutation, slicer_nonew)
                             if not is_droppedaxis(idx)]
        slicer_nodropped = [idx for idx in slicer if not is_droppedaxis(idx)]
        drop = [0 if idx is None else slice(None) for idx in slicer_nodropped]
        dat = dat[tuple(drop)]

        # 2) un-permute indices
        inv_map = invert_permutation(self.permutation)
        slicer_nonew = tuple(slicer_nonew[p] for p in inv_map)
        inv_map_nodropped = invert_permutation(dim_map_nodropped)
        dat = dat.transpose(inv_map_nodropped)

        # 3) byte-swap
        if ((sys.byteorder == 'little' and self.dtype.byteorder == '>') or
            (sys.byteorder == 'big' and self.dtype.byteorder == '<')):
            dat = dat.byteswap()

        # 4) write sub-array (defer to nibabel)
        if is_full(slicer_nonew, self._shape):
            # write the full array
            with self.fileobj('image') as f:
                array_to_file(dat, f, self.dtype,
                              offset=self._image.dataobj.offset,
                              order=self._image.dataobj.order)
        else:
            # write a slice into the existing array
            with self.fileobj('image') as f:
                writeslice(dat, f, tuple(slicer_nonew), self._shape, self.dtype,
                           offset=self._image.dataobj.offset,
                           order=self._image.dataobj.order,
                           heuristic=heuristic,
                           lock=self._lock['image'])

        # Finally) if compressed file + MGH -> write footer
        if self.is_compressed('header') and \
                hasattr(self._image.header, 'writeftr_to'):
            header = self._image.header
            header.set_data_dtype(self.dtype)
            header.set_data_shape(self.shape)
            with self.fileobj('header') as f:
                with self._lock['header']:
                    header.writeftr_to(f)

    def _data_raw(self):
        """Read native data

        Returns
        -------
        dat : np.ndarray[self.dtype]
            Sliced data loaded in memory.

        Notes
        -----
        .. The data can be memory mapped (this depends on nibabel)
        .. The data can be broadcasted (i.e., several values point to
           the same space in memory). To ensure that the data is
           contiguous, call `np.copy` on it.

        """

        # 1) remove new axes
        slicer = self.slicer or [slice(None)]*self.dim
        slicer_nonew = [idx for idx in slicer if not is_newaxis(idx)]
        dim_map_nodropped = [d for d, idx in zip(self.permutation, slicer_nonew)
                             if not is_droppedaxis(idx)]

        # 2) un-permute indices
        inv_map = invert_permutation(self.permutation)
        slicer_nonew = tuple(slicer_nonew[p] for p in inv_map)

        # 3) load sub-array (defer to nibabel)
        dat = self._image.dataobj._get_unscaled(slicer_nonew)

        # 4) permute
        dat = dat.transpose(dim_map_nodropped)

        # 5) compute output shape without any broadcasting
        shape_no_broadcast = []
        i = 0
        for idx in slicer:
            if is_droppedaxis(idx):
                continue
            else:
                if is_newaxis(idx):
                    shape_no_broadcast.append(1)
                else:  # is_sliceaxis(idx)
                    shape_no_broadcast.append(self.shape[i])
                i += 1
        dat = dat.reshape(shape_no_broadcast)

        # 6) broadcast
        # TODO: maybe this should be done *after* all the preprocessing?
        dat = np.broadcast_to(dat, self.shape)
        if not dat.flags.writeable:
            dat = np.copy(dat)

        return dat

    def _set_header_raw(self, header=None):
        """Write header to file

        Parameters
        ----------
        header : Header

        Returns
        -------
        self : type(self)

        """
        if header is None:
            header = self._image.dataobj._header
        header.set_data_dtype(self.dtype)
        header.set_data_shape(self.shape)
        if self.is_compressed('header'):
            # cannot write a chunk in the middle of the file
            return self._set_full_raw(header=header)
        with self.fileobj('header') as f:
            if hasattr(header, 'writehdr_to'):
                header.writehdr_to(f)
            if hasattr(header, 'writeftr_to'):
                header.writeftr_to(f)
            if hasattr(header, 'write_to'):
                f.seek(0)
                header.write_to(f)
        return self

    def _set_full_raw(self, header=None, dat=None):
        """Write the whole file to disk.

        This should only called when the data is compressed.

        Parameters
        ----------
        header : Header, optional
            If None, read from disk first
        dat : array_like, optional
            If None, read from disk first

        Returns
        -------
        self : type(self)

        """
        single_file = ('header' not in self._image.filemap
                       or self._image.filemap['image'] ==
                          self._image.filemap['header'])
        header_changed = header is not None
        dat_changed = dat is not None

        if not header_changed and not dat_changed:
            return self

        # load header in memory
        if header is None and single_file:
            header = self._image.dataobj._header
        header.set_data_dtype(self.dtype)
        header.set_data_shape(self.shape)

        # load data in memory
        if dat is None and single_file:
            dat = self._data_raw()

        # write everything
        if header is not None:
            with self.fileobj('header') as f:
                f.seek(0)
                if hasattr(header, 'writehdr_to'):
                    header.writehdr_to(f)
                elif hasattr(header, 'write_to'):
                    header.write_to(f)
        if dat is not None:
            self._set_data_raw(dat)
        if header is not None and hasattr(header, 'writeftr_to'):
            with self.fileobj('header') as f:
                header.writeftr_to(f)

        return self

    # ------------------------------------------------------------------
    #    HIGH-LEVEL IMPLEMENTATION
    # ------------------------------------------------------------------
    # These functions implement the MappedArray API
    # (that's why there's no doc, it's at the MappedArray level)

    def set_data(self, dat, casting='unsafe'):

        # --- convert to numpy ---
        dat = np.asanyarray(dat)
        info = cast_dtype.info(self.dtype)  # dtype on disk

        # --- cast ---
        if casting.startswith('rescale') and not info['is_floating_point']:
            # rescale
            # TODO: I am using float64 as an intermediate to cast
            #       Maybe I can do things in a nicer / more robust way
            minval = dat.min().astype(np.float64)
            maxval = dat.max().astype(np.float64)
            if dat.dtype != np.float64:
                dat = dat.astype(np.float64)
            if not dat.flags.writeable:
                dat = np.copy(dat)
            if casting == 'rescale':
                scale = (1 - minval/maxval) / (1 - info['min']/info['max'])
                offset = (info['max'] - info['min'])/(maxval - minval)
                dat *= scale
                dat += offset
            else:
                assert casting == 'rescale_zero'
                if minval < 0 and not info['is_signed']:
                    warn("Converting negative values to an unsigned datatype")
                scale = min(abs(info['max']/maxval) if maxval else float('inf'),
                            abs(info['min']/minval) if minval else float('inf'))
                dat *= scale

        # unsafe cast
        if dat.dtype != self.dtype:
            dat = dat.astype(self.dtype, casting='unsafe')

        # write on disk
        self._set_data_raw(dat)
        return self

    def data(self, dtype=None, device=None, casting='unsafe', rand=True,
             cutoff=None, dim=None, numpy=False):

        # --- sanity check before reading ---
        dtype = self.dtype if dtype is None else dtype
        info = cast_dtype.info(dtype)
        nptype = info['numpy']
        if not numpy and info['torch'] is None:
            raise TypeError('Data type {} does not exist in PyTorch.'
                            .format(dtype))

        # --- read native data ---
        dat = self._data_raw()
        info_in = cast_dtype.info(self.dtype)

        # --- cutoff ---
        if cutoff is not None:
            cutoff = sorted([100*val for val in pyutils.make_sequence(cutoff)])
            if len(cutoff) > 2:
                raise ValueError('Maximum to percentiles (min, max) should'
                                 ' be provided. Got {}.'.format(len(cutoff)))
            pct = np.nanpercentile(dat, cutoff, axis=dim, keepdims=True)
            if len(pct) == 1:
                dat = np.clip(dat, a_max=pct[0])
            else:
                dat = np.clip(dat, a_min=pct[0], a_max=pct[1])

        # --- cast ---
        scale = 1.
        if not casting.startswith('rescale'):
            # defer to numpy
            if nptype != dat.dtype:
                dat = dat.astype(nptype, casting=casting)

        elif not info['is_floating_point']:
            # rescale
            # TODO: I am using float64 as an intermediate to cast
            #       Maybe I can do things in a nicer / more robust way
            minval = dat.min().astype(np.float64)
            maxval = dat.max().astype(np.float64)
            if dat.dtype != np.float64:
                dat = dat.astype(np.float64)
            if not dat.flags.writeable:
                dat = np.copy(dat)
            if casting == 'rescale':
                scale = (1 - minval/maxval) / (1 - info['min']/info['max'])
                offset = (info['max'] - info['min'])/(maxval - minval)
                dat *= scale
                dat += offset
            else:
                assert casting == 'rescale_zero'
                if minval < 0 and not info['is_signed']:
                    warn("Converting negative values to an unsigned datatype")
                scale = min(abs(info['max']/maxval) if maxval else float('inf'),
                            abs(info['min']/minval) if minval else float('inf'))
                dat *= scale

        # --- random sample ---
        # uniform noise in the uncertainty interval
        info_tmp = cast_dtype.info(dat.dtype)
        if rand and info_in['is_integer'] and not info_tmp['is_integer']:
            noise = np.random.rand(*dat.shape).astype(dat.dtype)
            if scale != 1:
                noise *= scale
            dat += noise

        # unsafe cast
        if dat.dtype != nptype:
            dat = dat.astype(nptype, casting='unsafe')

        # convert to torch if needed
        if not numpy:
            dat = torch.as_tensor(dat, device=device)
        return dat

    def metadata(self, keys=None):
        if keys is None:
            keys = metadata_keys
        meta = header_to_metadata(self._image.dataobj._header, keys)
        return meta

    def set_metadata(self, **meta):
        header = self._image.dataobj._header
        if meta:
            header = metadata_to_header(header, meta)
        self._set_header_raw(header)

    # ------------------------------------------------------------------
    #    ADAPTED FROM NIBABEL'S ARRAYPROXY
    # ------------------------------------------------------------------
    # I am bypassing ArrayProxy to read and -- mainly -- write
    # data from/to disk. I've therefore copied the methods that deal
    # with file opening/permissions/locks.
    # Here, we want to write partially into existing files, so keeping
    # files open (including header files) is more common than in
    # vanilla nibabel

    @contextmanager
    def fileobj(self, key='image'):
        """Return a context manager that points to a file.

        It can be used with:
        >> nii = BabelArray('/path/to/file')
        >> with nii.fileobj('image') as f:
        >>    f.seek(pos)
        >>    ...

        """
        if key not in self._image.file_map:
            key = 'image'
        file_map = self._image.file_map[key]
        keep_open = self._keep_open[key]
        mode = self._mode + 'b'
        if self._keep_opener[key]:
            if not key in self._opener:
                self._opener[key] = file_map.get_prepare_fileobj(
                    keep_open=keep_open,
                    mode=mode)
            yield self._opener[key]
        else:
            with file_map.get_prepare_fileobj(
                    keep_open=False,
                    mode=mode) as opener:
                yield opener

    def __del__(self):
        """If this ``ArrayProxy`` was created with ``keep_file_open=True``,
        the open file object is closed if necessary.
        """
        for key, opener in self._opener:
            if not opener.closed:
                opener.close_if_mine()
            del self._opener[key]

    def __getstate__(self):
        """Returns the state of this ``ArrayProxy`` during pickling. """
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        """Sets the state of this ``ArrayProxy`` during unpickling. """
        self.__dict__.update(state)
        self._lock = RLock()


reader_classes.append(BabelArray)
