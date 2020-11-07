from .optionals import nibabel as nib
from ..core.optionals import numpy as np
from ..core import pyutils
from .file import MappedArray
from .indexing import invert_permutation
from . import dtype as cast_dtype
import torch
import os
from warnings import warn


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

    def __init__(self, file_like):
        """

        Parameters
        ----------
        file_like : str or fileobj
            Input file.
        """

        if nib is None:
            raise ImportError('NiBabel is not available.')

        if isinstance(file_like, (str, os.PathLike)):
            obj = nib.load(file_like)
        elif isinstance(file_like, nib.spatialimages.SpatialImage):
            obj = file_like
        else:
            raise TypeError('Input should be a filename or `SpatialImage` '
                            'object. Got {}.'.format(type(file_like)))

        self._image = obj

        super().__init__()

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
        slicer = [idx for idx in self.slicer if isinstance(idx, slice)
                  or (isinstance(idx, int) and idx >= 0)]
        perm = [p for p in self.perm if p is not None]

        # 2) un-permute indices
        iperm = invert_permutation(perm)
        slicer = tuple(slicer[p] for p in iperm)

        # 3) load sub-array (defer to nibabel)
        dat = self._image.dataobj._get_unscaled(slicer)

        # 4) permute
        dat = dat.transpose(perm)

        # 5) compute output shape without any broadcasting
        shape_no_broadcast = []
        i = 0
        for idx in self.slicer:
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

    def fdata(self, dtype=None, device=None, rand=False, cutoff=None,
              dim=None, numpy=False):

        # --- sanity check ---
        dtype = torch.get_default_dtype() if dtype is None else dtype
        info = cast_dtype.info(dtype)
        if not info['is_floating_point']:
            raise TypeError('Output data type should be a floating point '
                            'type but got {}.'.format(dtype))

        # --- get unscaled data ---
        dat = self.data(dtype=dtype, device=device, rand=rand,
                        cutoff=cutoff, dim=dim, numpy=numpy)

        # --- scale ---
        if self.slope != 1:
            dat *= self.slope
        if self.inter != 0:
            dat += self.inter

        return dat
