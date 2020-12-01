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
# nitorch imports
from nitorch.core import pyutils, dtypes
# local imports
from ..mapping import MappedArray, AccessType
from ..readers import reader_classes
from ..writers import writer_classes
from ..loadsave import map as map_array
from ..indexing import invert_permutation, is_newaxis, is_sliceaxis, \
                      is_droppedaxis, is_fullslice, split_operation
from .. import nputils
from ..metadata import keys as metadata_keys
from .metadata import header_to_metadata, metadata_to_header
from .utils import writeslice


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

    @property
    def readable(self):
        if self.is_compressed:
            return AccessType.Partial
        else:
            return AccessType.TruePartial

    @property
    def writable(self):
        if self.is_compressed:
            return AccessType.Partial
        else:
            return AccessType.TruePartial

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
        fileobj : nibabel.Opener, optional
            File object

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

        return fileslice(fileobj, slicer, self._shape, self.dtype,
                         offset=self._image.dataobj._offset,
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

        Dispatch to `_read_data_raw_full` or `_read_data_raw_partial`.

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
            if slicer is not None:
                dat = dat[slicer]
        else:
            dat = self._read_data_raw_partial(slicer, fileobj)

        return dat

    def _set_header_raw(self, header=None, fileobj=None):
        """Write (nibabel) header to file

        This function assumes that `fileobj` is at -- or can seek to --
        the beginning of the file.

        Parameters
        ----------
        header : nibabel.Header, default=`self._image.dataobj._header`
            Header object from nibabel.
        fileobj : nibabel.Opener, default=`self.fileobj('header')`
            File object., with `seek` and `write`.

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
        """Write (nibabel) footer to file

        This function assumes that `fileobj` is at -- or can seek to --
        the footer offset (that is, just after the data block).

        I think this is only used by the MGH format.

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
        dat = nputils.cast(dat, self.dtype, casting)

        # --- unpermute ---
        drop, perm, slicer = split_operation(self.permutation, self.slicer, 'w')
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
        dtype = dtypes.dtype(dtype)
        if not numpy and dtype.torch is None:
            raise TypeError('Data type {} does not exist in PyTorch.'
                            .format(dtype))

        # --- check that view is not empty ---
        if pyutils.prod(self.shape) == 0:
            if numpy:
                return np.zeros(self.shape, dtype=dtype.numpy)
            else:
                return torch.zeros(self.shape, dtype=dtype.torch, device=device)

        # --- read native data ---
        slicer, perm, newdim = split_operation(self.permutation, self.slicer, 'r')
        with self.fileobj('image', 'r') as f:
            dat = self._read_data_raw(slicer, fileobj=f)
        dat = dat.transpose(perm)[newdim]
        indtype = dtypes.dtype(self.dtype)

        # --- cutoff ---
        dat = nputils.cutoff(dat, cutoff, dim)

        # --- cast + rescale ---
        rand = rand and not indtype.is_floating_point
        tmpdtype = dtypes.float64 if (rand and not dtype.is_floating_point) else dtype
        dat, scale = nputils.cast(dat, tmpdtype.numpy, casting, with_scale=True)

        # --- random sample ---
        # uniform noise in the uncertainty interval
        if rand and not (scale == 1 and not dtype.is_floating_point):
            dat = nputils.addnoise(dat, scale)

        # --- final cast ---
        dat = nputils.cast(dat, dtype.numpy, 'unsafe')

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
        dtype = dtypes.dtype(dat.dtype)
        if not dtype.is_floating_point:
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
            header = format.header_class.from_header(like._image.dataobj._header)
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

        # set scale
        if hasattr(header, 'set_slope_inter'):
            slope, inter = header.get_slope_inter()
            if slope is None:
                slope = 1
            if inter is None:
                inter = 0
            header.set_slope_inter(slope, inter)

        # unscale
        if _savef:
            assert dtypes.dtype(dat.dtype).is_floating_point
            slope, inter = header.get_slope_inter()
            if inter not in (0, None) or slope not in (1, None):
                dat = dat.copy()
            if inter not in (0, None):
                dat -= inter
            if slope not in (1, None):
                dat /= slope

        # cast
        dat = nputils.cast(dat, dtype, casting)

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
        fhdr = fmap_header.get_prepare_fileobj('wb')
        if hasattr(header, 'writehdr_to'):
            header.writehdr_to(fhdr)
        elif hasattr(header, 'write_to'):
            header.write_to(fhdr)
        if fmap_image == fmap_header:
            fimg = fhdr
        else:
            fimg = fmap_image.get_prepare_fileobj('wb')
        array_to_file(dat, fimg, dtype,
                      offset=header.get_data_offset(),
                      order=image.ImageArrayProxy.order)
        if fmap_image == fmap_footer:
            fftr = fimg
        else:
            fftr = fmap_footer.get_prepare_fileobj('wb')
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
    # vanilla nibabel.

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
