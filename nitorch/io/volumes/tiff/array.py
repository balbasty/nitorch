# python
from contextlib import contextmanager
from warnings import warn
import torch
import numpy as np
# nitorch
from nitorch.spatial import affine_default
from nitorch.core import py, dtypes
# io
from nitorch.io.mapping import AccessType
from nitorch.io.utils.indexing import (is_fullslice, split_operation,
                                       slicer_sub2ind, invert_slice)
from nitorch.io.volumes.mapping import MappedArray
from nitorch.io.volumes.readers import reader_classes
from nitorch.io.volumes.writers import writer_classes
from nitorch.io.metadata import keys as metadata_keys
from nitorch.io.volumes.loadsave import map as map_array
# tiff
from tifffile import TiffFile, TiffFileError, TiffWriter
from .metadata import ome_zooms, parse_unit


class TiffArray(MappedArray):
    """
    MappedArray that uses `tifffile` under the hood.
    """

    FailedReadError = TiffFileError

    @classmethod
    def possible_extensions(cls):
        return '.tif', '.tiff'

    def __init__(self, file_like, mode='r', keep_open=True, **hints):
        """

        Parameters
        ----------
        file_like : str or file object
        mode : {'r'}, default='r'
        keep_open : bool, default=True
            Whether to keep the file handle open
        hints : keyword of the form `is_<format>=<True|False>`
            Tells the Tiff reader that a file is or isn't of a specific
            subformat. If not provided, it it guessed by the Tiff reader.
        """
        if mode not in ('r', 'rb'):
            raise ValueError('Tiff files can only be opened in mode "r"')
        if not isinstance(file_like, TiffFile):
            file_like = TiffFile(file_like, **hints)
            self._mine = True
        else:
            self._mine = False
        self._tiff = file_like
        if not keep_open:
            self._tiff.close()

        self._series = 0
        self._level = 0
        self._cache = dict()
        super().__init__()

    _series: int = 0   # index of series to map
    _level: int = 0    # index of pyramid level to map
    _cache: dict = {}  # a cache of precomputed _shape, _spatial, etc

    def close_if_mine(self):
        if self._mine:
            self._tiff.close()
        super().close_if_mine()
        return self

    def close(self):
        self._tiff.close()
        super().close()
        return self

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
        msk = [ax in 'XYZ' for ax in self._axes]
        return msk

    @property
    def _affine(self):
        """Affine orientation matrix of a series+level"""
        # TODO: I don't know yet how we should use GeoTiff to encode
        #   affine matrices. In the matrix/zooms, their voxels are ordered
        #   as [x, y, z] even though their dimensions in the returned array
        #   are ordered as [Z, Y, X]. If we want to keep the same convention
        #   as nitorch, I need to permute the matrix/zooms.
        if '_affine' not in self._cache:
            with self.tiffobj() as tiff:
                omexml = tiff.ome_metadata
                geotags = tiff.geotiff_metadata or {}
            zooms, units, axes = ome_zooms(omexml, self.series)
            if zooms:
                # convert to mm + drop non-spatial zooms
                units = [parse_unit(u) for u in units]
                zooms = [z * (f / 1e-3) for z, (f, type) in zip(zooms, units)
                         if type in ('m', 'pixel')]
                if 'ModelPixelScaleTag' in geotags:
                    warn("Both OME and GeoTiff pixel scales are present: "
                         "{} vs {}. Using OME."
                         .format(zooms, geotags['ModelPixelScaleTag']))
            elif 'ModelPixelScaleTag' in geotags:
                zooms = geotags['ModelPixelScaleTag']
                axes = 'XYZ'
            else:
                zooms = 1.
                axes = [ax for ax in self._axes if ax in 'XYZ']
            if 'ModelTransformation' in geotags:
                aff = geotags['ModelTransformation']
                aff = torch.as_tensor(aff, dtype=torch.double).reshape(4, 4)
                self._cache['_affine'] = aff
            elif ('ModelTiepointTag' in geotags):
                # copied from tifffile
                sx, sy, sz = py.make_list(zooms, n=3)
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
                zooms = py.make_list(zooms, n=len(axes))
                ax2zoom = {ax: zoom for ax, zoom in zip(axes, zooms)}
                axes = [ax for ax in self._axes if ax in 'XYZ']
                shape = [shp for shp, msk in zip(self._shape, self._spatial)
                         if msk]
                zooms = [ax2zoom.get(ax, 1.) for ax in axes]
                layout = [('R' if ax == 'Z' else 'P' if ax == 'Y' else 'S')
                          for ax in axes]
                aff = affine_default(shape, zooms, layout=''.join(layout))
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

    @property
    def readable(self):
        # That's not exact: pseudo partial access in-plane
        return AccessType.TruePartial

    @property
    def writable(self):
        return AccessType.No

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

    def __del__(self):
        # make sure we close all file objects
        if hasattr(self, '_tiff') and hasattr(self._tiff, 'close'):
            self._tiff.close()

    @property
    def filename(self):
        with self.tiffobj() as f:
            return f.filename

    _tiff_kinds = (
        'shaped',
        'bigtiff',
        'lsm',
        'mmstack',
        'ome',
        'imagej',
        'ndtiff',
        'fluoview',
        'stk',
        'sis',
        'svs',
        'scn',
        'qpi',
        'ndpi',
        'bif',
        'scanimage',
        'nih',
        'mdgel',
        'uniform',
    )

    _metadata_types = (
        'shaped',
        'ome',
        'scn',
        'lsm',
        'stk',
        'imagej',
        'fluoview',
        'nih',
        'fei',
        'sem',
        'sis',
        'mdgel',
        'andor',
        'epics',
        'metaseries',
        'pilatus',
        'micromanager',
        'gdal',
        'scanimage',
        'geotiff',
        'gdal',
        'astrotiff',
        'streak',
        'eer',
    )

    def _info4writer(self):
        info = {}
        with self.tiffobj() as tiff:
            info['byteorder'] = tiff.byteorder
            for kind in self._tiff_kinds:
                info['is_' + kind] = getattr(tiff, 'is_' + kind, None)
            for metadata_dtype in self._metadata_types:
                info[metadata_dtype + '_metadata'] = getattr(tiff, metadata_dtype + '_metadata', None)
            info['dtype'] = tiff.pages.first.dtype
            # shaped = [samplesperpixel_separate, imagedepth, imagelength, imagewidth, samplesperpixel_contig]
            info['shaped'] = tiff.pages.first.shaped
            info['axes'] = tiff.pages.first.axes  # "SXYZ"
            info['photometric'] = tiff.pages.first.photometric
            info['compression'] = tiff.pages.first.compression
            info['predictor'] = tiff.pages.first.predictor
            info['subsampling'] = tiff.pages.first.subsampling
            info['colormap'] = tiff.pages.first.colormap
            info['description'] = tiff.pages.first.description
            info['datetime'] = tiff.pages.first.datetime
            info['resolution'] = tiff.pages.first.resolution
            info['resolutionunit'] = tiff.pages.first.resolutionunit
        return info

    def raw_data(self):
        # --- check that view is not empty ---
        if py.prod(self.shape) == 0:
            return np.zeros(self.shape, dtype=self.dtype)

        # --- read native data ---
        slicer, perm, newdim = split_operation(self.permutation, self.slicer, 'r')
        with self.tiffobj() as f:
            dat = self._read_data_raw(slicer, tiffobj=f)
        dat = dat.transpose(perm)[newdim]
        return dat

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

        # build header
        init_kwargs = {}
        write_kwargs = {}
        if isinstance(like, TiffArray):
            info = like._info4writer()
            default = lambda key: info[key] or None
        else:
            info = None
            default = lambda key: None
        init_kwargs['ome'] = metadata.get('ome', default('is_ome'))
        init_kwargs['imagej'] = metadata.get('imagej', default('is_imagej'))
        init_kwargs['bigtiff'] = metadata.get('bigtiff', default('is_bigtiff'))
        # init_kwargs['shaped'] = metadata.get('shaped', default('is_shaped'))
        init_kwargs['byteorder'] = metadata.get('byteorder', default('byteorder'))
        write_kwargs['photometric'] = metadata.get('photometric', default('photometric') or 'MINISBLACK')
        write_kwargs['compression'] = metadata.get('compression', default('compression'))
        write_kwargs['predictor'] = metadata.get('predictor', default('predictor'))
        write_kwargs['subsampling'] = metadata.get('subsampling', default('subsampling'))
        write_kwargs['colormap'] = metadata.get('colormap', default('colormap'))
        write_kwargs['description'] = metadata.get('description', default('description'))
        write_kwargs['datetime'] = metadata.get('datetime', default('datetime'))
        # if 'affine' in metadata:
        #     vx = metadata['affine'].cpu().detach().square().sum(0).sqrt().numpy()
        #     write_kwargs['resolution'] = 1/vx
        #     write_kwargs['resolutionunit'] = metadata.get('voxel_size_unit', 'mm')
        # else:
        #     write_kwargs['resolution'] = metadata.get('resolution', default('resolution'))
        #     write_kwargs['resolutionunit'] = metadata.get('resolutionunit', default('resolutionunit'))
        if info:
            if init_kwargs['ome']:
                write_kwargs['metadata'] = info['ome_metadata'] or None
            elif init_kwargs['imagej']:
                write_kwargs['metadata'] = info['imagej_metadata'] or None
            elif init_kwargs['shaped']:
                write_kwargs['metadata'] = info['shaped_metadata'] or None
            else:
                for metadata_type in cls._metadata_types:
                    if info.get(metadata_type + '_metadata', None):
                        write_kwargs['metadata'] = info[metadata_type + '_metadata']
                        break

        if init_kwargs['bigtiff'] is None:
            init_kwargs['bigtiff'] = (
                dat.nbytes > 2 ** 32 - 2 ** 25
                and not init_kwargs['imagej']
                and write_kwargs['compression'] in (None, 0, 1, 'NONE', 'none')
            )

        dat = dat.squeeze()

        # write everything
        with TiffWriter(file_like, **init_kwargs) as writer:
            writer.write(dat, dtype=dtype, **write_kwargs)
            
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
                        else idx for idx, shp in zip(slicer_stack, shape_stack)]

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
                count != py.prod(page.shape) * page.bitspersample // 8
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
        if channels > 1 and (py.prod(shape) if shape else 1) != images:
            shape.append(channels)
            axes.append('C')

        remain = images // (py.prod(shape) if shape else 1)
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
writer_classes.append(TiffArray)
