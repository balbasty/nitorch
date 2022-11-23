from .mapping import MappedArray
from ..mapping import AccessType
from ..utils.opener import open, Opener, transform_opener, is_compressed_fileobj
from ..utils.indexing import is_fullslice, splitop
from ..utils.volumeio import read_array, write_array, read_subarray, write_subarray
from ..utils import volutils
from ..loadsave import map as map_array
from nitorch.core import py, dtypes
import numpy as np
from threading import RLock
from contextlib import contextmanager
from pathlib import Path
import torch


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
    _filemap: dict = {}                 # Map of all input files

    def is_compressed(self, key=None):
        with self.fileobj(key) as f:
            if isinstance(f, Opener):
                f = f.fileobj
            iscomp = is_compressed_fileobj(f)
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
        drop, perm, slicer = splitop(self.permutation, self.slicer, 'w')
        dat = dat[drop].transpose(perm)

        # --- dispatch ---
        if self.is_compressed('data'):
            if not is_fullslice(slicer, self._shape):
                # read-and-write
                slice = dat
                dat = self._read_data_raw()
                dat[slicer] = slice
            with self.fileobj('data', 'w', seek=0) as f:
                if self.same_file('data', 'header'):
                    self._write_header_raw(fileobj=f)
                self._write_data_raw_full(dat, fileobj=f)
                if self.same_file('data', 'footer'):
                    self._write_footer_raw(fileobj=f)

        elif is_fullslice(slicer, self._shape):
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
        slicer, perm, newdim = splitop(self.permutation, self.slicer, 'r')
        with self.fileobj('image', 'r') as f:
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

        self.to_header(**meta)

        if self.is_compressed('header'):
            # read-and-write
            if self.same_file('data', 'header'):
                with self.fileobj('header', 'r') as f:
                    full_dat = self._read_data_raw(fileobj=f)
            with self.fileobj('header', 'w') as f:
                self._write_header_raw(fileobj=f)
                if self.same_file('header', 'data'):
                    self._write_data_raw_full(full_dat, fileobj=f)
                if self.same_file('header', 'footer'):
                    self._write_footer_raw(fileobj=f)
        else:
            with self.fileobj('header', 'r+') as f:
                self._write_header_raw(fileobj=f)

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
            header = getattr(like._image.dataobj, '_header',
                             like._image.header)
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
        array_to_file(dat, fimg, dtype,
                      offset=header.get_data_offset(),
                      order=image.ImageArrayProxy.order)
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
    #    LOW-LEVEL IMPLEMENTATION
    # ------------------------------------------------------------------
    # the following functions implement read/write of (meta)data at the
    # lowest level (no conversion is performed there)

    def _write_data_raw_partial(self, dat, slicer, fileobj=None, key=None):
        key = key or list(self._filemap.keys())[0]

        if fileobj is None:
            # Create our own context
            with self.fileobj(key, 'r+') as fileobj:
                return self._write_data_raw_partial(dat, slicer, fileobj, key)

        # sanity checks for developers
        # (asserts because proper conversion/dispatch should happen before)
        assert isinstance(dat, np.ndarray), "Data should already be numpy"
        assert dat.dtype == self.dtype, "Data should already have correct type"
        assert not self.is_compressed(key), "Data cannot be compressed"
        assert not is_fullslice(slicer, self._shape), "No need for partial writing"

        if not fileobj.readable():
            raise RuntimeError('File object not readable')
        if not fileobj.writable():
            raise RuntimeError('File object not writable')

        # write data chunk
        write_subarray(dat, fileobj, tuple(slicer), self._shape, self.dtype,
                       offset=self._offset_dat,
                       order=self._order,
                       lock=self._lock[key])

    def _write_data_raw_full(self, dat, fileobj=None, key=None):
        key = key or list(self._filemap.keys())[0]

        if fileobj is None:
            # Create our own context
            with self.fileobj(key, 'a') as fileobj:
                return self._write_data_raw_full(dat, fileobj, key)

        # sanity checks for developers
        # (asserts because proper conversion/dispatch should happen before)
        assert isinstance(dat, np.ndarray), "Data should already be numpy"
        assert dat.dtype == self.dtype, "Data should already have correct type"

        if not fileobj.writable():
            raise RuntimeError('File object not writable')

        write_array(dat, fileobj, self.dtype,
                    offset=self._offset_dat, order=self._order)

    def _read_data_raw_partial(self, slicer, fileobj=None, lock=None, key=None):
        key = key or list(self._filemap.keys())[0]
        lock = lock or self._lock[key]

        if fileobj is None:
            with self.fileobj(key, 'r') as fileobj:
                return self._read_data_raw_partial(slicer, fileobj, lock, key)

        return read_subarray(fileobj, slicer, self._shape, self.dtype,
                             offset=self._offset_dat,
                             order=self._order,
                             lock=lock)

    def _read_data_raw_full(self, fileobj=None, lock=None, key=None):
        key = key or list(self._filemap.keys())[0]
        lock = lock or self._lock[key]

        if fileobj is None:
            with self.fileobj(key, 'r') as fileobj:
                return self._read_data_raw_full(fileobj, lock)

        return read_array(fileobj, self._shape, self.dtype, fileobj,
                          offset=self._offset_dat,
                          order=self._order)

    def _read_data_raw(self, slicer=None, fileobj=None):
        if not slicer or is_fullslice(slicer, self._shape):
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
                        self._opener[(key, mode)] = open(file_like, mode,
                                                         keep_open=True)
                    except ValueError:
                        self._opener[(key, mode)] = open(file_like, 'rb',
                                                         keep_open=True)
                else:
                    self._opener[(key, 'r')] = open(file_like, 'rb',
                                                    keep_open=False)
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

