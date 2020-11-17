"""Implementation of MappedArray based on nibabel.

..warning:: This file assumes that nibabel (and numpy) is available and
            should not be imported if it is not the case!

This implementation is a mess. It works well for formats that hold
the data in a contiguous manner and have the metadata in a single
(or eventually) two other contiguous blocks, such as MGH or
Analyze-derived formats (including Nifti). However, it fails with other
nibabel formats (MINC, PARREC, ...). I am a bit lost between following
my abstraction and trying to not reimplement nibabel entirely...

As of now (Nov 2020), reading and writing Nifti and MGH works. And I
still believe that my MappedArray API is cleaner.
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
from nibabel.volumeutils import _is_compressed_fobj as is_compressed_fobj, \
                                array_from_file, array_to_file
from nibabel.fileslice import fileslice
from nibabel.filebasedimages import ImageFileError
from nibabel.fileholders import FileHolder
# nitorch imports
from nitorch.core import pyutils
# local imports
from .mapping import MappedArray
from .readers import reader_classes
from .writers import writer_classes
from .loadsave import map as map_array
from .indexing import invert_permutation, is_newaxis, is_sliceaxis, \
                      is_droppedaxis, is_fullslice
from . import dtype as cast_dtype
from .metadata import keys as metadata_keys
from ._babel_metadata import header_to_metadata, metadata_to_header
from ._babel_utils import writeslice


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
            for one_ext in klass.valid_exts:
                ext.append(one_ext)
        return tuple(ext)

    FailedReadError = ImageFileError

    # ------------------------------------------------------------------
    #    ATTRIBUTES
    # ------------------------------------------------------------------
    # A bunch of attributes are known by nibabel. I just link to them
    # when that's the case

    fname = property(lambda self: self._image.file_map['image'].filename)
    _affine = property(lambda self: torch.as_tensor(self._image.affine, dtype=torch.double))
    _spatial = property(lambda self: tuple([True]*3 + [False]*max(0, self._dim-3)))
    _shape = property(lambda self: [int(d) for d in self._image.shape])
    dtype = property(lambda self: self._image.dataobj.dtype)
    slope = property(lambda self: float(self._image.dataobj.slope))
    inter = property(lambda self: float(self._image.dataobj.inter))

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

    def _write_data_raw_partial(self, dat, slicer, fileobj=None):
        """Write native data

        Parameters
        ----------
        dat : np.ndarray
            Should already have the on-disk data type, including byte-order
        slicer : tuple[index_like]
            Unpermuted slicer, without new axes.

        """

        if fileobj is None:
            # Create our own context
            with self.fileobj('image') as fileobj:
                return self._write_data_raw_partial(dat, slicer, fileobj)

        # sanity checks for developers
        # (asserts because proper conversion/dispatch should happen before)
        assert isinstance(dat, np.ndarray), "Data should already be numpy"
        assert dat.dtype == self.dtype, "Data should already have correct type"
        assert not self.is_compressed('image'), "Data cannot be compressed"
        assert not all(is_fullslice(slicer, self._shape)), "No need for partial writing"

        if not fileobj.fobj.readable():
            raise RuntimeError('File object not readable')
        if not fileobj.fobj.writable():
            raise RuntimeError('File object not writable')

        # write data chunk
        writeslice(dat, fileobj, tuple(slicer), self._shape, self.dtype,
                   offset=self._image.dataobj.offset,
                   order=self._image.dataobj.order,
                   lock=self._lock['image'])

    def _write_data_raw_full(self, dat, fileobj=None):
        """Write native data

        Returns
        -------
        dat : np.ndarray
            Should already have the on-disk data type, including byte-order

        """

        if fileobj is None:
            # Create our own context
            with self.fileobj('image') as fileobj:
                return self._write_data_raw_full(dat, fileobj)

        # sanity checks for developers
        # (asserts because proper conversion/dispatch should happen before)
        assert isinstance(dat, np.ndarray), "Data should already be numpy"
        assert dat.dtype == self.dtype, "Data should already have correct type"

        if not fileobj.fobj.writable():
            raise RuntimeError('File object not writable')

        array_to_file(dat, fileobj, self.dtype,
                      offset=self._image.dataobj.offset,
                      order=self._image.dataobj.order)

    def _read_data_raw_partial(self, slicer, fileobj=None, lock=None):
        """Read a chunk of data from disk

        Parameters
        ----------
        slicer : tuple[index_like]
        fileobj : Opener
        lock : Lock

        Returns
        -------
        dat : np.ndarray

        """
        if fileobj is None:
            with self.fileobj('image') as fileobj:
                return self._read_data_raw_partial(slicer, fileobj, lock)

        if lock is None:
            lock = self._lock['image']
            return self._read_data_raw_partial(slicer, fileobj, lock)

        return fileslice(fileobj, slicer,
                         self._shape,
                         self.dtype,
                         self._image.dataobj._offset,
                         order=self._image.dataobj.order,
                         lock=lock)

    def _read_data_raw_full(self, fileobj=None, lock=None, mmap=None):
        """Read the full data from disk

        Parameters
        ----------
        fileobj : Opener, default=`self.fileobj('image')`
        lock : Lock, default=`self._lock['image']`
        mmap : bool, default=`self._image.dataobj._mmap`

        Returns
        -------
        dat : np.ndarray

        """
        if fileobj is None:
            with self.fileobj('image') as fileobj:
                return self._read_data_raw_full(fileobj, lock)

        if lock is None:
            with self._lock['image'] as lock:
                return self._read_data_raw_full(fileobj, lock)

        if mmap is None:
            mmap = self._image.dataobj._mmap

        return array_from_file(self._shape, self.dtype, fileobj,
                               offset=self._image.dataobj._offset,
                               order=self._image.dataobj.order,
                               mmap=mmap)

    def _read_data_raw(self, slicer=None, fileobj=None, mmap=None):
        """Read native data

        Parameters
        ----------
        slicer : tuple[index_like], optional
            A tuple of indices that describe the chunk of data to read.
            If None, read everything.
        fileobj : file object, default=`self.fileobj('image', 'r')`
            A file object (with `seek`, `read`) from which to read
        mmap : bool, default=`self._image.dataobj.mmap`
            If True, try to memory map the data.

        Returns
        -------
        dat : np.ndarray

        """

        if fileobj is None:
            with self.fileobj('image', 'r') as f:
                return self._read_data_raw(fileobj=f)

        # load sub-array
        if slicer is None or all(is_fullslice(slicer, self._shape)):
            dat = self._read_data_raw_full(fileobj, mmap=mmap)
        else:
            dat = self._read_data_raw_partial(slicer, fileobj)

        return dat

    def _set_header_raw(self, header=None, fileobj=None):
        """Write header to file

        Parameters
        ----------
        header : Header
        fileobj : Opener

        Returns
        -------
        self : type(self)

        """
        if fileobj is None:
            with self.fileobj('header') as f:
                return self._set_header_raw(header, f)
        if header is None:
            header = self._image.dataobj._header
        header.set_data_dtype(self.dtype)
        header.set_data_shape(self._shape)
        if hasattr(header, 'writehdr_to'):
            header.writehdr_to(fileobj)
        elif hasattr(header, 'write_to'):
            fileobj.seek(0)
            header.write_to(fileobj)
        return self

    def _set_footer_raw(self, header=None, fileobj=None):
        """Write footer to file

        Parameters
        ----------
        header : Header
        fileobj : Opener

        Returns
        -------
        self : type(self)

        """
        if fileobj is None:
            with self.fileobj('footer') as f:
                return self._set_footer_raw(header, f)
        if header is None:
            header = self._image.dataobj._header
        header.set_data_dtype(self.dtype)
        header.set_data_shape(self._shape)
        if hasattr(header, 'writeftr_to'):
            header.writeftr_to(fileobj)
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
            dat = self._read_data_raw()

        # write everything
        if header is not None:
            with self.fileobj('header') as f:
                f.seek(0)
                if hasattr(header, 'writehdr_to'):
                    header.writehdr_to(f)
                elif hasattr(header, 'write_to'):
                    header.write_to(f)
        if dat is not None:
            self._write_data_raw_partial(dat)
        if header is not None and hasattr(header, 'writeftr_to'):
            with self.fileobj('header') as f:
                header.writeftr_to(f)

        return self

    @staticmethod
    def _split_op(perm, slicer, direction):
        """Split the operation `slicer of permutation` into subcomponents.

        Symbolic slicing is encoded by a permutation and an indexing operation.
        The operation `sub.data()` where `sub` has been obtained by applying
        a succession of permutations and sub-indexings on `full` should be
        equivalent to
        ```python
        >>> full_data = full.data()
        >>> sub_data = full_data.permute(sub.permutation)[sub.slicer]
        ```
        However, the slicer may create new axes or drop (i.e., index with
        a scalar) original axes. This function splits `slicer of permutation`
        into more sub-components.
        If the direction is 'read'
            * `slicer_sub`  : unpermuted slicer without new axes
            * `perm`        : permutation without dropped axes
            * `slicer_add`  : slicer that adds new axes
        ```python
        >>> dat = full[slicer_sub].transpose(perm)[slicer_add]
        ```
        If the direction is 'write':
            * `slicer_drop` : slicer that drops new axes
            * `perm`        : inverse permutation without dropped axes
            * `slicer_sub`  : unpermuted slicer without
        ```python
        >>> full[slicer_sub] = dat[slicer_drop].transpose(perm)
        ```

        Parameters
        ----------
        perm : sequence[int]
            Permutation of `range(self._dim)`
        slicer : sequence[index_like]
            Sequence of indices that slice into `self._shape`
        direction : {'r', 'w'}
            Either split for reading ('r') or writing ('w') a chunk.

        Returns
        -------
        slicer_sub ('r') or slicer_drop ('w') : tuple
        perm : tuple
        slicer_add ('r') or slicer_sub ('w') : tuple

        """
        def remap(perm):
            """Re-index dimensions after some have been dropped"""
            remaining_dims = sorted(perm)
            dim_map = {}
            for new, old in enumerate(remaining_dims):
                dim_map[old] = new
            remapped_dim = [dim_map[d] for d in perm]
            return remapped_dim

        def select(index, seq):
            """Select elements into a sequence using a list of indices"""
            return [seq[idx] for idx in list(index)]

        slicer_nonew = list(filter(lambda x: not is_newaxis(x), slicer))
        slicer_nodrop = filter(lambda x: not is_droppedaxis(x), slicer)
        slicer_sub = select(invert_permutation(perm), slicer_nonew)
        perm_nodrop = [d for d, idx in zip(perm, slicer_nonew)
                       if not is_droppedaxis(idx)]
        perm_nodrop = remap(perm_nodrop)

        if direction.lower().startswith('r'):
            slicer_add = map(lambda x: slice(None) if x is not None else x,
                             slicer_nodrop)
            return tuple(slicer_sub), tuple(perm_nodrop), tuple(slicer_add)
        elif direction.lower().startswith('w'):
            slicer_drop = map(lambda x: 0 if x is None else slice(None),
                              slicer_nodrop)
            inv_perm_nodrop = invert_permutation(perm_nodrop)
            return tuple(slicer_drop), tuple(inv_perm_nodrop), tuple(slicer_sub)
        else:
            raise ValueError("direction should be in ('read' ,'write') "
                             "but got {}.".format(direction))

    @staticmethod
    def _cast(dat, dtype, casting='unsafe'):
        """

        Parameters
        ----------
        dat : np.ndarray
        dtype : np.dtype
        casting : str, defualt='unsafe'

        Returns
        -------
        dat : np.ndarray[dtype]

        """
        info = cast_dtype.info(dtype)
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
            casting = 'unsafe'

        # unsafe cast
        if dat.dtype != dtype:
            dat = dat.astype(dtype, casting=casting)

        return dat

    # ------------------------------------------------------------------
    #    HIGH-LEVEL IMPLEMENTATION
    # ------------------------------------------------------------------
    # These functions implement the MappedArray API
    # (that's why there's no doc, it's at the MappedArray level)

    def set_data(self, dat, casting='unsafe'):

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
        dat = self._cast(dat, self.dtype, casting)

        # --- unpermute ---
        drop, perm, slicer = self._split_op(self.permutation, self.slicer, 'w')
        dat = dat[drop].transpose(perm)

        # --- dispatch ---
        if self.is_compressed('image'):
            if not all(is_fullslice(slicer, self._shape)):
                # read-and-write
                slice = dat
                with self.fileobj('image', 'r') as f:
                    dat = self._read_data_raw(fileobj=f, mmap=False)
                dat[slicer] = slice
            with self.fileobj('image', 'w', seek=0) as f:
                if self.same_file('image', 'header'):
                    self._set_header_raw(fileobj=f)
                self._write_data_raw_full(dat, fileobj=f)
                if self.same_file('image', 'footer'):
                    self._set_footer_raw(fileobj=f)

        elif all(is_fullslice(slicer, self._shape)):
            with self.fileobj('image', 'r+') as f:
                self._write_data_raw_full(dat, fileobj=f)

        else:
            with self.fileobj('image', 'r+') as f:
                self._write_data_raw_partial(dat, slicer, fileobj=f)
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

        # --- check that view is not empty ---
        if pyutils.prod(self.shape) == 0:
            if numpy:
                return np.zeros(self.shape, dtype=info['numpy'])
            else:
                return torch.zeros(self.shape, dtype=info['torch'], device=device)

        # --- read native data ---
        slicer, perm, newdim = self._split_op(self.permutation, self.slicer, 'r')
        with self.fileobj('image', 'r') as f:
            dat = self._read_data_raw(slicer, fileobj=f)
        dat = dat.transpose(perm)[newdim]
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

        if self.is_compressed('header'):
            # read-and-write
            if self.same_file('image', 'header'):
                with self.fileobj('header', 'r') as f:
                    full_dat = self._read_data_raw(fileobj=f)
            with self.fileobj('header', 'w') as f:
                self._set_header_raw(header, fileobj=f)
                if self.same_file('header', 'image'):
                    self._write_data_raw_full(full_dat, fileobj=f)
                if self.same_file('header', 'footer'):
                    self._set_footer_raw(header, fileobj=f)
        else:
            with self.fileobj('header', 'r+') as f:
                self._set_header_raw(header, fileobj=f)

    @classmethod
    def savef_new(cls, dat, file_like, like=None, **metadata):
        # sanity check
        info = cast_dtype.info(dat.dtype)
        if not info['is_floating_point']:
            raise TypeError('Input data type should be a floating point '
                            'type but got {}.'.format(dat.dtype))

        # detach
        if torch.is_tensor(dat):
            dat = dat.detach().cpu().numpy()

        # defer
        cls.save_new(dat, file_like, like=like, casting='unsafe',
                     _savef=True, **metadata)

    @classmethod
    def save_new(cls, dat, file_like, like=None, casting='unsafe',
                 _savef=False, **metadata):

        if torch.is_tensor(dat):
            dat = dat.detach().cpu()
        dat = np.asanyarray(dat)
        if like is not None:
            like = map_array(like)

        # guess data type:
        def guess_dtype():
            dtype = None
            if dtype is None:
                dtype = metadata.get('dtype', None)
            if dtype is None and like is not None:
                dtype = like.dtype
            if dtype is None:
                dtype = dat.dtype
            return dtype
        dtype = guess_dtype()

        def guess_format():
            # 1) from extension
            ok_klasses = []
            if isinstance(file_like, str):
                base, ext = os.path.splitext(file_like)
                if ext.lower() == '.gz':
                    base, ext = os.path.splitext(base)
                ok_klasses = [klass for klass in all_image_classes
                              if ext in klass.valid_exts]
                if len(ok_klasses) == 1:
                    return ok_klasses[0]
            # 2) from like
                if isinstance(like, BabelArray):
                    return type(like._image)
            # 3) from extension (if conflict)
            if len(ok_klasses) != 0:
                return ok_klasses[0]
            # 4) fallback to nifti-1
            return nib.Nifti1Image
        format = guess_format()

        # build header
        if isinstance(like, BabelArray):
            # defer metadata conversion to nibabel
            header = format.header_class(like._image.header)
        else:
            header = format.header_class()
            if like is not None:
                # copy generic metadata
                like_metadata = like.metadata()
                like_metadata.update(metadata)
                metadata = like_metadata
        header = metadata_to_header(header, metadata)

        # check endianness
        disk_byteorder = header.endianness
        data_byteorder = dtype.byteorder
        if disk_byteorder == '=':
            disk_byteorder = '<' if sys.byteorder == 'little' else '>'
        if data_byteorder == '=':
            data_byteorder = '<' if sys.byteorder == 'little' else '>'
        if disk_byteorder != data_byteorder:
            dtype = dtype.newbyteorder()

        # unscale
        if _savef:
            assert cast_dtype.info(dat.dtype)['is_floating_point']
            slope, inter = header.get_slope_inter()
            if inter not in (0, None) or slope not in (1, None):
                dat = dat.copy()
            if inter not in (0, None):
                dat -= inter
            if slope not in (1, None):
                dat /= slope

        # cast
        dat = cls._cast(dat, dtype, casting)

        # set dtype / shape
        header.set_data_dtype(dtype)
        header.set_data_shape(dat.shape)

        # create image object
        image = format(dat, affine=None, header=header)

        # write everything
        file_map = format.filespec_to_file_map(file_like)
        fmap_header = file_map.get('header', file_map.get('image'))
        fmap_image = file_map.get('image')
        fmap_footer = file_map.get('footer', file_map.get('image'))
        fhdr = fmap_header.get_prepare_fileobj('w')
        if hasattr(header, 'writehdr_to'):
            header.writehdr_to(fhdr)
        elif hasattr(header, 'write_to'):
            header.write_to(fhdr)
        if fmap_image == fmap_header:
            fimg = fhdr
        else:
            fimg = fmap_image.get_prepare_fileobj('w')
        array_to_file(dat, fimg, dtype,
                      offset=header.get_data_offset(),
                      order=image.ImageArrayProxy.order)
        if fmap_image == fmap_footer:
            fftr = fimg
        else:
            fftr = fmap_footer.get_prepare_fileobj('w')
        if hasattr(header, 'writeftr_to'):
            header.writeftr_to(fftr)

    # ------------------------------------------------------------------
    #    ADAPTED FROM NIBABEL'S ARRAYPROXY
    # ------------------------------------------------------------------
    # I am bypassing ArrayProxy to read and -- mainly -- write
    # data from/to disk. I've therefore copied the methods that deal
    # with file opening/permissions/locks.
    # Here, we want to write partially into existing files, so keeping
    # files open (including header files) is more common than in
    # vanilla nibabel

    def same_file(self, *keys):
        filenames = [self.filename(key) for key in keys]
        return len(set(filenames)) == 1

    def filename(self, key='image'):
        if key not in self._image.file_map:
            key = 'image'
        return self._image.file_map[key].filename

    def filemap(self, key='image'):
        if key not in self._image.file_map:
            key = 'image'
        return self._image.file_map[key]

    @contextmanager
    def fileobj(self, key='image', permission='', seek=None):
        """Return a `nibabel.openers.Opener`.

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
        mode = (permission or self._mode) + 'b'
        if self._keep_opener[key]:
            if key in self._opener:
                # check that the permission fits what we want
                f = self._opener[key].fobj
                mode_ok = ((not mode.startswith('r') or f.readable()) and
                           (not mode.startswith(('w', 'r+')) or f.writable()))
                if not mode_ok:
                    if not self._opener[key].closed:
                        self._opener[key].close_if_mine()
                    del self._opener[key]
            if not key in self._opener or self._opener[key].closed:
                self._opener[key] = file_map.get_prepare_fileobj(
                    keep_open=keep_open,
                    mode=mode)
            if seek is not None:
                # check that the cursor position is correct
                try:
                    self._opener[key].seek(seek)
                except OSError:
                    # we're in a gzipped file and trying to perform
                    # a negative seek. Better to close and reopen.
                    self._opener[key].close_if_mine()
                    self._opener[key] = file_map.get_prepare_fileobj(
                        keep_open=keep_open,
                        mode=mode)
                    self._opener[key].seek(seek)
            yield self._opener[key]
        else:
            with file_map.get_prepare_fileobj(keep_open=False, mode=mode) as f:
                if seek is not None:
                    f.seek(seek)
                yield f

    def __del__(self):
        """If this ``ArrayProxy`` was created with ``keep_file_open=True``,
        the open file object is closed if necessary.
        """
        if hasattr(self, '_opener'):
            for key, opener in self._opener.items():
                if not opener.closed:
                    opener.close_if_mine()
            del self._opener

    def __getstate__(self):
        """Returns the state of this ``ArrayProxy`` during pickling. """
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        """Sets the state of this ``ArrayProxy`` during unpickling. """
        self.__dict__.update(state)
        self._lock = dict()
        for key in self._image.file_map:
            self._lock[key] = RLock()


reader_classes.append(BabelArray)
writer_classes.append(BabelArray)
