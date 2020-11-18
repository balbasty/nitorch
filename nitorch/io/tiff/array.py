from ..mapping import MappedArray
from ..indexing import is_fullslice, split_operation, slicer_sub2ind, invert_slice
from .. import dtype as cast_dtype
from .. import nputils
from ..readers import reader_classes
from nitorch.spatial import affine_default
from nitorch.core import pyutils
from tifffile import TiffFile
from contextlib import contextmanager
import torch
import numpy as np


class TiffArray(MappedArray):
    """
    MappedArray that uses `tifffile` under the hood.
    """

    def __init__(self, file_like, keep_file_open=True, **hints):
        """

        Parameters
        ----------
        file_like : str or file object
        keep_file_open : bool, default=True
            Whether to keep the file handle open
        hints : keyword of the form `is_<format>=<True|False>`
            Tells the Tiff reader that a file is or isn't of a specific
            subformat. If not provided, it it guessed by the Tiff reader.
        """
        self._tiff = TiffFile(file_like, **hints)
        if not keep_file_open:
            self._tiff.close()

        super().__init__()

    _series: int = 0   # index of series to map
    _level: int = 0    # index of pyramid level to map
    _cache: dict = {}  # a cache of precomputed _shape, _spatial, etc

    @property
    def _shape(self):
        """Full shape of a series+level"""
        if '_shape' not in self._cache:
            with self.tiffobj() as tiff:
                shape = tiff.series[self.series].levels[self.level].shape
            self._cache['_shape'] = shape
        return self._cache['_shape']

    @property
    def _axes(self):
        """Axes names of a series+level"""
        if '_axes' not in self._cache:
            with self.tiffobj() as tiff:
                axes = tiff.series[self.series].levels[self.level].axes
            self._cache['_axes'] = axes
        return self._cache['_axes']

    @property
    def _spatial(self):
        """Mask of spatial axes of a series+level"""
        msk = [ax in ('X', 'Y', 'Z') for ax in self._axes]
        return msk

    @property
    def _affine(self):
        """Affine orientation matrix of a series+level"""
        # TODO: read vx from OME/ImageJ metadata
        if '_affine' not in self._cache:
            with self.tiffobj() as tiff:
                geotags = tiff.geotiff_tags() or {}
            if 'ModelTransformation' in geotags:
                aff = geotags['ModelTransformation']
                aff = torch.as_tensor(aff, dtype=torch.double).reshape(4, 4)
                self._cache['_affine'] = aff
            elif 'ModelPixelScaleTag' in geotags and 'ModelTiepointTag' is geotags:
                # copied from tifffile
                sx, sy, sz = geotags['ModelPixelScaleTag']
                tiepoints = torch.as_tensor(geotags['ModelTiepointTag'])
                affines = []
                for tiepoint in tiepoints:
                    i, j, k, x, y, z = tiepoint
                    affines.append(torch.as_tensor(
                        [[sx,  0.0, 0.0, x - i * sx],
                         [0.0, -sy, 0.0, y + j * sy],
                         [0.0, 0.0, sz,  z - k * sz],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.double))
                affines = torch.stack(affines, dim=0)
                if len(tiepoints) == 1:
                    affines = affines[0]
                    self._cache['_affine'] = affines
            else:
                zooms = geotags.get('ModelPixelScaleTag', [1., 1., 1.])
                shape = [shp for shp, msk in zip(self._shape, self._spatial)
                         if msk]
                zooms = pyutils.make_list(zooms, len(shape))
                aff = affine_default(shape, zooms, layout='RPS')
                self._cache['_affine'] = aff
        return self._cache['_affine']


    @property
    def dtype(self):
        if 'dtype' not in self._cache:
            with self.tiffobj() as tiff:
                dt = tiff.series[self.series].levels[self.level].dtype
            self._cache['dtype'] = dt
        return self._cache['dtype']

    @property
    def series(self):
        """Series index (Tiff files can hold multiple series)"""
        return self._series

    @series.setter
    def series(self, val):
        if val != self.series and not all(is_fullslice(self.slicer)):
            raise RuntimeError("Cannot change series in a view")
        self._series = val
        self._cache = {}

    @property
    def level(self):
        """Level index (Tiff files can hold multiple spatial resolutions)"""
        return self._level

    @level.setter
    def level(self, val):
        if val != self.level and not all(is_fullslice(self.slicer)):
            raise RuntimeError("Cannot change resolution level in a view")
        self._level = val
        self._cache = {}

    @contextmanager
    def tiffobj(self):
        """Returns an *open* Tiff reader.

        Should be used in a `with` statement:
        ```python
        >>> with self.tiffobj() as tiff:
        >>>     # do stuff with `tiff`
        ```
        """
        closed = self._tiff.filehandle.closed
        if closed:
            self._tiff.filehandle.open()
        yield self._tiff
        if closed:
            self._tiff.close()

    @property
    def filename(self):
        with self.tiffobj() as f:
            return f.filename

    def data(self, dtype=None, device=None, casting='unsafe', rand=True,
             cutoff=None, dim=None, numpy=False):

        # --- sanity check before reading ---
        dtype = self.dtype if dtype is None else dtype
        outinfo = cast_dtype.info(dtype)
        if not numpy and outinfo['torch'] is None:
            raise TypeError('Data type {} does not exist in PyTorch.'
                            .format(dtype))
        if not isinstance(dtype, np.dtype):
            dtype = outinfo['numpy']

        # --- check that view is not empty ---
        if pyutils.prod(self.shape) == 0:
            if numpy:
                return np.zeros(self.shape, dtype=dtype)
            else:
                return torch.zeros(self.shape, dtype=outinfo['torch'], device=device)

        # --- read native data ---
        slicer, perm, newdim = split_operation(self.permutation, self.slicer, 'r')
        with self.tiffobj() as f:
            dat = self._read_data_raw(slicer, tiffobj=f)
        dat = dat.transpose(perm)[newdim]
        ininfo = cast_dtype.info(self.dtype)

        # --- cutoff ---
        dat = nputils.cutoff(dat, cutoff, dim)

        # --- cast ---
        rand = rand and ininfo['is_integer']
        if rand and not outinfo['is_floating_point']:
            tmpdtype = np.float64
        else:
            tmpdtype = dtype
        dat, scale = nputils.cast(dat, tmpdtype, casting, with_scale=True)

        # --- random sample ---
        # uniform noise in the uncertainty interval
        if rand and not (scale == 1 and outinfo['is_integer']):
            noise = np.random.rand(*dat.shape)
            if scale != 1:
                noise *= scale
            noise = noise.astype(dat.dtype)
            dat += noise

        # --- final cast ---
        dat = nputils.cast(dat, dtype, 'unsafe')

        # convert to torch if needed
        if not numpy:
            dat = torch.as_tensor(dat, device=device)
        return dat

    # --------------
    #   LOW LEVEL
    # --------------

    def _read_data_raw(self, slicer=None, tiffobj=None):
        """Read native data

        Dispatch to `_read_data_raw_full` or `_read_data_raw_partial`.

        Parameters
        ----------
        slicer : tuple[index_like], optional
            A tuple of indices that describe the chunk of data to read.
            If None, read everything.
        tiffobj : file object, default=`self.fileobj('image', 'r')`
            A file object (with `seek`, `read`) from which to read

        Returns
        -------
        dat : np.ndarray

        """
        if tiffobj is None:
            with self.tiffobj() as tiffobj:
                return self._read_data_raw(slicer, tiffobj)

        # load sub-array
        if slicer is None or all(is_fullslice(slicer, self._shape)):
            dat = self._read_data_raw_full(tiffobj)
        else:
            dat = self._read_data_raw_partial(slicer, tiffobj)

        return dat

    def _read_data_raw_partial(self, slicer, tiffobj=None):
        """Read a chunk of data from disk

        Parameters
        ----------
        slicer : tuple[slice or int]
        tiffobj : TiffFile

        Returns
        -------
        dat : np.ndarray

        """
        if tiffobj is None:
            with self.tiffobj() as tiffobj:
                return self._read_data_raw_partial(slicer, tiffobj)

        # 1) split dimensions
        shape_feat, shape_stack, shape_page = self._shape_split(tiffobj)
        dim_feat = len(shape_feat)
        dim_stack = len(shape_stack)
        dim_page = len(shape_page)

        # 2) split slicer
        slicer_feat = slicer[:dim_feat]
        slicer_stack = slicer[dim_feat:dim_feat+dim_stack]
        slicer_page = slicer[dim_feat+dim_stack:]

        dim_feat_out = sum(isinstance(idx, slice) for idx in slicer_feat)
        dim_stack_out = sum(isinstance(idx, slice) for idx in slicer_stack)
        dim_page_out = sum(isinstance(idx, slice) for idx in slicer_page)

        # 3) ensure positive strides
        slicer_inv = [slice(None, None, -1) if idx.step and idx.step < 0
                      else slice(None) for idx in slicer_stack
                      if isinstance(idx, slice)]
        slicer_stack = [invert_slice(idx, shp) if isinstance(idx, slice) and
                                                  idx.step and idx.step < 0
                        else idx for idx, shp in zip(slicer_stack, shape_stack)
                        if isinstance(idx, slice)]

        # 4) convert stack slice to list of linear indices
        #    (or to one slice if possible)
        index_stack = slicer_sub2ind(slicer_stack, shape_stack)

        # 5) read only pages in the substack
        dat = tiffobj.asarray(key=index_stack,
                              series=self.series,
                              level=self.level)
        dat = dat.reshape([*shape_feat, -1, *shape_page])

        # 6) apply slicers along the feature and page dimensions
        dat = dat[(*slicer_feat, slice(None), *slicer_page)]

        # 7) reshape
        dat = dat.reshape(self.shape)

        # 7) final slicers for negative strides along stack dimensions
        slicer = [slice(None)] * dim_feat_out + slicer_inv + [slice(None)] * dim_page_out
        dat = dat[tuple(slicer)]

        return dat

    def _read_data_raw_full(self, tiffobj=None):
        """Read the full data from disk

        Parameters
        ----------
        tiffobj : TiffFile

        Returns
        -------
        dat : np.ndarray

        """
        if tiffobj is None:
            with self.tiffobj() as tiffobj:
                return self._read_data_raw_full(tiffobj)

        return tiffobj.asarray(series=self.series, level=self.level)

    def _shape_split(self, tiffobj=None):
        """Split the shape into different components

        Returns
        -------
        shape_feat : tuple[int]
            Color features (belong to pages but end-up at the left-most axis)
        shape_collection : tuple[int]
            Shape of the collection of pages (usually Z, T, etc. axes)
        shape_page : tuple[int]
            Shape of one page -- with or without features (usually X, Y axes)
        """
        if tiffobj is None:
            with self.tiffobj() as tiffobj:
                return self._shape_split(tiffobj)

        if tiffobj.is_imagej:
            return self._shape_split_imagej(tiffobj)
        else:
            page = tiffobj.series[self.series].levels[self.level].pages[0]
            shape_page = page.shape
            page_dim = len(shape_page)
            shape_collection = self._shape[:-page_dim]
            return tuple(), tuple(shape_collection), tuple(shape_page)

    def _shape_split_imagej(self, tiffobj):
        """Split the shape into different components (ImageJ format).

        This is largely copied from tifffile.
        """

        pages = tiffobj.pages
        pages.useframes = True
        pages.keyframe = 0
        page = pages[0]
        meta = tiffobj.imagej_metadata

        def is_virtual():
            # ImageJ virtual hyperstacks store all image metadata in the first
            # page and image data are stored contiguously before the second
            # page, if any
            if not page.is_final:
                return False
            images = meta.get('images', 0)
            if images <= 1:
                return False
            offset, count = page.is_contiguous
            if (
                count != pyutils.prod(page.shape) * page.bitspersample // 8
                or offset + count * images > self.filehandle.size
            ):
                raise ValueError()
            # check that next page is stored after data
            if len(pages) > 1 and offset + count * images > pages[1].offset:
                return False
            return True

        isvirtual = is_virtual()
        if isvirtual:
            # no need to read other pages
            pages = [page]
        else:
            pages = pages[:]

        images = meta.get('images', len(pages))
        frames = meta.get('frames', 1)
        slices = meta.get('slices', 1)
        channels = meta.get('channels', 1)

        # compute shape of the collection of pages
        shape = []
        axes = []
        if frames > 1:
            shape.append(frames)
            axes.append('T')
        if slices > 1:
            shape.append(slices)
            axes.append('Z')
        if channels > 1 and (pyutils.prod(shape) if shape else 1) != images:
            shape.append(channels)
            axes.append('C')

        remain = images // (pyutils.prod(shape) if shape else 1)
        if remain > 1:
            shape.append(remain)
            axes.append('I')

        if page.axes[0] == 'S' and 'C' in axes:
            # planar storage, S == C, saved by Bio-Formats
            return tuple(), tuple(shape), tuple(page.shape[1:])
        elif page.axes[0] == 'I':
            # contiguous multiple images
            return tuple(), tuple(shape), tuple(page.shape[1:])
        elif page.axes[:2] == 'SI':
            # color-mapped contiguous multiple images
            return tuple(page.shape[0:1]), tuple(shape), tuple(page.shape[2:])
        else:
            return tuple(), tuple(shape), tuple(page.shape)


reader_classes.append(TiffArray)
