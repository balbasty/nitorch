from .optionals import nibabel as nib
from ..core.optionals import numpy as np
from ..core import pyutils
from .file import MappedArray
from .indexing import invert_permutation
from . import dtype as cast_dtype
from ._babel_utils import writeslice, is_full, array_to_file
from nibabel import openers
import torch
import os
from warnings import warn
from contextlib import contextmanager


@contextmanager
def _get_fileobj(self):
    """Create and return a new ``ImageOpener``, or return an existing one.

    The specific behaviour depends on the value of the ``keep_file_open``
    flag that was passed to ``__init__``.

    Yields
    ------
    ImageOpener
        A newly created ``ImageOpener`` instance, or an existing one,
        which provides access to the file.
    """
    if self._persist_opener:
        if not hasattr(self, '_opener'):
            self._opener = openers.ImageOpener(
                self.file_like,
                keep_open=self._keep_file_open,
                mode=self._mode)
        yield self._opener
    else:
        with openers.ImageOpener(
                self.file_like,
                keep_open=False,
                mode=self._mode) as opener:
            yield opener


class BabelArray(MappedArray):
    """MappedArray that relies on NiBabel."""


    fname = property(lambda self: self._image.file_map['image'])
    fileobj = property(lambda self: self._image.dataobj._get_fileobj())
    _affine = property(lambda self: torch.as_tensor(self._image.affine))
    _spatial = property(lambda self: tuple([True]*3 + [False]*max(0, self._dim-3)))
    _shape = property(lambda self: self._image.shape)
    dtype = property(lambda self: self._image.dataobj.dtype)
    slope = property(lambda self: self._image.dataobj.slope)
    inter = property(lambda self: self._image.dataobj.inter)

    @property
    def is_compressed(self):
        # TODO: it's not perfect -- in a two-file system we could have
        #       the image compressed but not the header
        #       we could have a function that takes the file tag as
        #       argument
        with self.fileobj as f:
            if isinstance(f, nib.openers.Opener):
                f = f.fobj
            iscomp = nib.volumeutils._is_compressed_fobj(f)
        return iscomp

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

        if isinstance(file_like, (str, os.PathLike)):
            obj = nib.load(file_like, mmap=False, keep_file_open=keep_open)
        elif isinstance(file_like, nib.spatialimages.SpatialImage):
            obj = file_like
        else:
            raise TypeError('Input should be a filename or `SpatialImage` '
                            'object. Got {}.'.format(type(file_like)))

        obj.dataobj._mode = perm + 'b'
        import types
        obj.dataobj._get_fileobj = types.MethodType(_get_fileobj, obj.dataobj)
        self._image = obj

        super().__init__()

    def _set_data_raw(self, dat):
        """Write native data"""
        # 0.a) convert to numpy
        if torch.is_tensor(dat):
            dat = dat.detach().cpu().numpy()
        dat = np.asanyarray(dat)

        # 0.b) sanity check for broadcasting
        slicer = self.slicer or [slice(None)]*self.dim
        if any(isinstance(idx, int) and idx < 0 for idx in slicer):
            raise ValueError('Cannot write into a broadcasted volume.')

        # 1) remove new axes
        drop = [0 if idx is None else slice(None) for idx in slicer]
        slicer = [idx for idx in slicer if isinstance(idx, slice)
                  or (isinstance(idx, int) and idx >= 0)]
        permutation = [p for p in self.permutation if p is not None]
        dat = dat[tuple(drop)]

        # 2) un-permute indices
        ipermutation = invert_permutation(permutation)
        slicer = tuple(slicer[p] for p in ipermutation)
        dat = dat.transpose(ipermutation)

        # 3) write sub-array (defer to nibabel)
        if is_full(slicer, self._shape):
            with self.fileobj as f:
                array_to_file(dat, f, self.dtype,
                              offset=self._image.dataobj.offset,
                              order=self._image.dataobj.order)
        else:
            with self.fileobj as f:
                writeslice(dat, f, slicer, self._shape, self.dtype,
                           offset=self._image.dataobj.offset,
                           order=self._image.dataobj.order,
                           lock=self._image.dataobj._lock)

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
        permutation = self.permutation
        slicer_simple = [idx for idx in slicer if isinstance(idx, slice)
                         or (isinstance(idx, int) and idx >= 0)]
        permutation = [p for p in permutation if p is not None]

        # 2) un-permute indices
        ipermutation = invert_permutation(permutation)
        slicer_simple = tuple(slicer_simple[p] for p in ipermutation)

        # 3) load sub-array (defer to nibabel)
        dat = self._image.dataobj._get_unscaled(slicer_simple)

        # 4) permute
        dat = dat.transpose(permutation)

        # 5) compute output shape without any broadcasting
        shape_no_broadcast = []
        i = 0
        for idx in slicer:
            if isinstance(idx, int) and idx >= 0:
                continue
            else:
                if idx is None or isinstance(idx, int):
                    shape_no_broadcast.append(1)
                else:
                    shape_no_broadcast.append(self.shape[i])
                i += 1
        dat = dat.reshape(shape_no_broadcast)

        # 6) broadcast
        # TODO: maybe this should be done *after* all the preprocessing?
        dat = np.broadcast_to(dat, self.shape)
        if not dat.flags.writeable:
            dat = np.copy(dat)

        return dat

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
