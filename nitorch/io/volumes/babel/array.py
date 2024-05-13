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
from contextlib import contextmanager
from threading import RLock
# external imports
import torch
import numpy as np
import nibabel as nib
# nibabel imports
from nibabel.spatialimages import SpatialImage
from nibabel.imageclasses import all_image_classes
from nibabel.volumeutils import _is_compressed_fobj as is_compressed_fobj, \
                                array_from_file, array_to_file
from nibabel.fileslice import fileslice
from nibabel.filebasedimages import ImageFileError
# nitorch imports
from nitorch.core import py, dtypes
# io imports
from nitorch.io.mapping import AccessType
from nitorch.io.volumes.mapping import MappedArray
from nitorch.io.volumes.readers import reader_classes
from nitorch.io.volumes.writers import writer_classes
from nitorch.io.volumes.loadsave import map as map_array
from nitorch.io.utils.indexing import is_fullslice, split_operation
from nitorch.io.utils.opener import open, Opener, transform_opener, gz
from nitorch.io.utils import volutils
from nitorch.io.metadata import keys as metadata_keys
from .metadata import header_to_metadata, metadata_to_header
from .utils import writeslice

# With x, y, z being quaternions, NiBabel's reader requires that:
# 1 - [x, y, z]*[x, y, z]' > threshold, where threshold = -1e-7 (by
# default - see nibabel.quaternions.fillpositive - the dot product is
# supposed to be 1, but this sometimes does not hold because of numerical
# precision). As this is a bit conservative, we here decreases this
# threshold.
nib.Nifti1Header.quaternion_threshold = -1e-6


class BabelArray(MappedArray):
    """MappedArray that relies on NiBabel."""

    # register gzip opener for mgz files
    Opener.ext_map['.mgz'] = gz

    def __init__(self, file_like, mode='r', keep_open=True):
        """

        Parameters
        ----------
        file_like : str or fileobj
            Input file.
        mode : {'r', 'r+'}, default='r'
            File permission
        keep_open : bool, default=True
            Keep file descriptor open.
        """

        if nib is None:
            raise ImportError('NiBabel is not available.')

        if isinstance(file_like, SpatialImage):
            self._image = file_like
        else:
            self._image = nib.load(file_like, mmap=False, keep_file_open=False)

        # deal with file openers
        if not mode in ('r', 'r+'):
            raise ValueError(f"Mode expected in ('r', 'r+'). Got {mode}.")
        self.mode = mode            # Decides if the user lets us write
        self.keep_open = keep_open  # Keep file descriptor open (user value)?
        self._prepare_openers()

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
    dtype = property(lambda self: self._image.get_data_dtype())
    slope = property(lambda self: float(getattr(self._image.dataobj, 'slope', None)))
    inter = property(lambda self: float(getattr(self._image.dataobj, 'inter', None)))

    @slope.setter
    def slope(self, val):
        self._image.dataobj.slope = float(np.asarray(val).item())

    @inter.setter
    def inter(self, val):
        self._image.dataobj.inter = float(np.asarray(val).item())

    def is_compressed(self, key='image'):
        with self.fileobj(key) as f:
            if isinstance(f, Opener):
                f = f.fileobj
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
            with self.fileobj('image', 'r+') as fileobj:
                return self._write_data_raw_partial(dat, slicer, fileobj)

        # sanity checks for developers
        # (asserts because proper conversion/dispatch should happen before)
        assert isinstance(dat, np.ndarray), "Data should already be numpy"
        assert dat.dtype == self.dtype, "Data should already have correct type"
        assert not self.is_compressed('image'), "Data cannot be compressed"
        assert not all(is_fullslice(slicer, self._shape)), "No need for partial writing"

        if not fileobj.readable():
            raise RuntimeError('File object not readable')
        if not fileobj.writable():
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
            with self.fileobj('image', 'a') as fileobj:
                return self._write_data_raw_full(dat, fileobj)

        # sanity checks for developers
        # (asserts because proper conversion/dispatch should happen before)
        assert isinstance(dat, np.ndarray), "Data should already be numpy"
        assert dat.dtype == self.dtype, "Data should already have correct type"

        if not fileobj.writable():
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
            with self.fileobj('image', 'r') as fileobj:
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
            with self.fileobj('image', 'r') as fileobj:
                return self._read_data_raw_full(fileobj, lock)

        if lock is None:
            with self._lock['image'] as lock:
                return self._read_data_raw_full(fileobj, lock)

        if mmap is None:
            mmap = getattr(self._image.dataobj, '_mmap', False)

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

        if not isinstance(self._image, (nib.MGHImage, nib.AnalyzeImage)):
            # Use nibabel's high-level API
            image = self._image
            if slicer is not None:
                image = image.slicer[slicer]
            return np.asanyarray(image.dataobj)

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
        fileobj : Opener, default=`self.fileobj('header')`
            File object., with `seek` and `write`.
            By default, a file object open in mode 'w' (and therefore
            truncated)

        Returns
        -------
        self : type(self)

        """
        if fileobj is None:
            with self.fileobj('header', 'w') as f:
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
            By default, a file object open in mode 'a'

        Returns
        -------
        self : type(self)

        """
        if fileobj is None:
            with self.fileobj('footer', 'a') as f:
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

    def raw_data(self):
        # --- check that view is not empty ---
        if py.prod(self.shape) == 0:
            return np.zeros(self.shape, dtype=self.dtype)

        # --- read native data ---
        slicer, perm, newdim = split_operation(self.permutation, self.slicer, 'r')
        with self.fileobj('image', 'r') as f:
            dat = self._read_data_raw(slicer, fileobj=f)
        dat = dat.transpose(perm)[newdim]
        return dat

    def metadata(self, keys=None):
        if not keys:
            keys = metadata_keys
        header = getattr(self._image.dataobj, '_header', self._image.header)
        meta = header_to_metadata(header, keys)
        return meta

    def set_metadata(self, **meta):
        if '+' not in self.mode:
            raise RuntimeError('Cannot write into read-only volume. '
                               'Re-map in mode "r+" to allow in-place '
                               'writing.')

        header = self._image.dataobj._header
        if meta:
            header = metadata_to_header(header, meta, shape=self.shape)

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

        if isinstance(dat, MappedArray):
            if like is None:
                like = dat
            dat = dat.fdata(numpy=True)

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

        if isinstance(dat, MappedArray):
            if like is None:
                like = dat
            dat = dat.data(numpy=True)
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
            dtype = dtypes.dtype(dtype).numpy
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
                if len(ok_klasses) > 0:
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
            header = getattr(like._image.dataobj, '_header', like._image.header)
            header = format.header_class.from_header(header)
        else:
            header = format.header_class()
        if like is not None:
            # copy generic metadata
            like_metadata = like.metadata()
            like_metadata.update(metadata)
            metadata = like_metadata
        # set shape now so that we can set zooms/etc
        header.set_data_shape(dat.shape)
        header = metadata_to_header(header, metadata, shape=dat.shape)

        # check endianness
        disk_byteorder = header.endianness
        data_byteorder = dtype.byteorder
        if disk_byteorder == '=':
            disk_byteorder = '<' if sys.byteorder == 'little' else '>'
        if data_byteorder == '=':
            data_byteorder = '<' if sys.byteorder == 'little' else '>'
        if disk_byteorder != data_byteorder:
            dtype = dtype.newbyteorder()

        # get scale
        slope, inter = header.get_slope_inter()
        if slope is None:
            slope = 1
        if inter is None:
            inter = 0

        # unscale
        if _savef:
            assert dtypes.dtype(dat.dtype).is_floating_point
            if inter not in (0, None) or slope not in (1, None):
                dat = dat.copy()
            if inter not in (0, None):
                dat -= inter
            if slope not in (1, None):
                dat /= slope
        # cast + setdtype
        dat, s, o = volutils.cast(dat, dtype, casting=casting,
                                  returns='dat+scale+offset')
        header.set_data_dtype(dat.dtype)

        # set scale
        if hasattr(header, 'set_slope_inter'):
            slope = slope / s
            inter = inter - slope * o
            if slope in (0, float('inf'), -float('inf'), float('nan')):
                slope = None
            if inter in (0, float('inf'), -float('inf'), float('nan')):
                inter = 0
            header.set_slope_inter(slope, inter)

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
        default_order = getattr(
            image.ImageArrayProxy, '_default_order',
            getattr(image.ImageArrayProxy, 'order', 'F'))
        array_to_file(dat, fimg, dtype,
                      offset=header.get_data_offset(),
                      order=default_order)
        if fmap_image == fmap_footer:
            fftr = fimg
        else:
            fftr = fmap_footer.get_prepare_fileobj('wb')
        if hasattr(header, 'writeftr_to'):
            header.writeftr_to(fftr)

        # close files
        if not fhdr.closed:
            fhdr.close_if_mine()
        if not fimg.closed:
            fimg.close_if_mine()
        if not fftr.closed:
            fftr.close_if_mine()

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

    def _prepare_openers(self):
        # Saved openers
        self._opener = dict()
        # Locks (still unsure why I need that)
        self._lock = dict()

        for key, val in self._image.file_map.items():
            mode = self.mode + 'b'  # everything is binary in nibabel (?)
            file_like = val.file_like
            self._lock[key] = RLock()
            if hasattr(file_like, 'read') and hasattr(file_like, 'seek'):
                # file object -> keep stuff irrelevant
                if not file_like.readable() or not file_like.seekable():
                    raise ValueError('File must be readable and seekable')
                if '+' in self.mode and not file_like.writable():
                    raise ValueError('File must be writable in mode "r+"')
                self._opener[(key, self.mode)] = file_like
                continue
            try:
                if self.keep_open:
                    try:
                        self._opener[(key, mode)] = open(file_like, mode, keep_open=True)
                    except ValueError:
                        self._opener[(key, mode)] = open(file_like, 'rb', keep_open=True)
                else:
                    self._opener[(key, 'r')] = open(file_like, 'rb', keep_open=False)
                    if not self._opener[(key, 'r')].is_indexed:
                        del self._opener[(key, 'r')]
            except FileNotFoundError:
                continue

    @contextmanager
    def fileobj(self, key='image', mode='', seek=None):
        """Return an `Opener`.

        It can be used with:
        >> nii = BabelArray('/path/to/file')
        >> with nii.fileobj('image') as f:
        >>    f.seek(pos)
        >>    ...

        Parameters
        ----------
        key : str, default='image'
            Key of the file to open
        mode : {'r', 'r+', 'w', 'w+', 'a', 'a+'}, default='r'
            Opening mode. The file type ('b' or 't') should be omitted.
        seek : int, optional
            Position to seek.

        """
        if key not in self._image.file_map:
            key = 'image'
        if not mode:
            mode = 'r'
        opener = None
        # check if we have the right opener
        if (key, mode) in self._opener:
            opener = self._opener[(key, mode)]
        # otherwise, check if we have one with more permissions than needed
        if opener is None and (key, mode + '+') in self._opener:
            opener = self._opener[(key, mode + '+')]
        # otherwise, check if we can hack a liberal one
        # (e.g., 'r+' -> 'w(+)' or 'a(+)')
        if opener is None:
            for (key0, mode0), opener0 in self._opener.items():
                if key0 != key:
                    continue
                try:
                    opener = transform_opener(opener0, mode)
                    break
                except ValueError:
                    pass
        # we found one -> perform a few sanity checks
        if opener is not None:
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
        file_map = self._image.file_map[key]
        with open(file_map.file_like, mode=mode, keep_open=False) as opener:
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
        self._lock = dict()
        for key in self._image.file_map:
            self._lock[key] = RLock()


reader_classes.append(BabelArray)
writer_classes.append(BabelArray)
