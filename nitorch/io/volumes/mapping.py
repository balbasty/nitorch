from copy import copy
import torch
from nitorch.core.py import make_list
from nitorch.core import dtypes
from nitorch.spatial import affine_sub, affine_permute, voxel_size as affvx
from nitorch.io.utils.indexing import (expand_index, guess_shape, compose_index, neg2pos,
                                       is_droppedaxis, is_newaxis, is_sliceaxis,
                                       invert_permutation, invert_slice, slice_navigator)
from ..utils import volutils
from ..mapping import MappedFile


class MappedArray(MappedFile):
    """Base class for mapped arrays.

    Mapped arrays are usually stored on-disk, along with (diverse) metadata.

    They can be symbolically sliced, allowing for partial reading and
    (sometimes) writing of data from/to disk.
    Chaining of symbolic slicing operations is implemented in this base
    class. The actual io must be implemented by the child class.

    Abstract Methods
    ----------------

    Child classes MUST implement:
    * self.data(...)

    Child classes SHOULD implement:
    * self.metadata(...)         default -> returns empty dict

    Child classes MAY implement:
    * self.set_data(...)         default -> raises cls.FailedWriteError
    * self.set_metadata(...)     default -> raises cls.FailedWriteError
    * cls.save_new(...)          default -> raises cls.FailedWriteError
    * cls.savef_new(...)         default -> raises cls.FailedWriteError

    Child classes SHOULD register themselves in `readers.reader_classes`.
    If they implement `save_new`, child classes SHOULD register
    themselves in `writers.writer_classes`.

    Properties
    ----------
    dtype : np.dtype            On-disk data type
    slope : float               Intensity slope from on-disk to unit
    inter : float               Intensity shift from on-disk to unit
    affine : tensor             Orientation matrix: maps spatial axes to 'world'
    spatial : tuple[bool]       Mask of 'spatial' axes (x, y, z, ...)
    slicer : tuple[index_like]  Indexing into the full on-disk array
    permutation : tuple[int]    Permutation of the original in-disk axes.
    dim : int                   Number of axes
    voxel_size : tuple[float]   World size of the spatial dimensions
    readable : AccessType       See `AccessType`
    writable : AccessType       See `AccessType`

    Types
    -----
    FailedReadError             Error raised when failing to load
    FailedWriteError            Error raised when failing to save

    Methods
    -------
    slice(tuple[index_like])    Subslice the array
    permute(tuple[int])         Permute axes
    transpose(int, int)         Permute two axes
    unsqueeze(int)              Insert singleton dimension
    squeeze(int)                Remove singleton dimension
    unbind -> tuple             Unstack arrays along a dimension
    chunk -> tuple              Unstack arrays along a dimension by chunks
    split -> tuple              Unstack arrays along a dimension by chunks

    data(...) -> tensor         Load raw data to memory
    fdata(...) -> tensor        Load scaled floating-point data to memory
    metadata(...) -> dict       Load metadata to memory
    set_data(dat, ...)          Write raw data to disk
    set_fdata(dat, ...)         Write scaled floating-point data to disk
    set_metadata(**meta)        Write metadata to disk

    Class methods
    -------------
    save_new(dat, file_like)    Write new file populated with `dat`
    savef_new(dat, file_like)   Write new file populated with (scaled) `dat`

    External functions
    ------------------
    map(file_like) -> MappedArray   Build a MappedArray
    load(file_like) -> tensor       Load raw data to memory from a file
    loadf(file_like) -> tensor      Load scaled data to memory from a file
    save(dat, file_like) ->         Save raw data into a new file
    savef(dat, file_like) ->        Save scaled data into a new file
    cat(tuple[MappedArray])         Concatenate arrays along a dimension

    Syntaxic sugar
    --------------
    __call__    -> fdata        Load scaled floating-point data to memory
    __array__   -> fdata        Load scaled floating-point data to memory
    __getitem__ -> slice        Subslice the array
    __setitem__ -> set_fdata    Write scaled floating-point data to disk
    __len__                     Size of the first dimension (or 0 if scalar)

    """

    fname: str = None             # filename (can be None if in-memory proxy)
    fileobj = None                # file-like object (`write`, `seek`, etc)
    is_compressed: bool = None    # is compressed
    dtype: torch.dtype = None     # on-disk data type
    slope: float = 1              # intensity slope
    inter: float = 0              # intensity shift

    affine = None                 # sliced voxel-to-world
    _affine = None                # original voxel-to-world
    spatial: tuple = None         # sliced spatial mask (len -> dim)
    _spatial: tuple = None        # original spatial mask (len -> _dim)
    shape: tuple = None           # sliced shape (len -> dim)
    _shape: tuple = None          # original shape (len -> _dim)
    slicer: tuple = None          # indexing into the parent
    permutation: tuple = None     # permutation of original dim (len -> _dim)

    dim = property(lambda self: len(self.shape))    # Nb of sliced dimensions
    _dim = property(lambda self: len(self._shape))  # Nb of original dimensions
    voxel_size = property(lambda self: affvx(self.affine))

    def __init__(self, **kwargs):
        self._init(**kwargs)

    def _init(self, **kwargs):
        for key, val in kwargs:
            setattr(self, key, val)

        if self.permutation is None:
            self.permutation = tuple(range(self._dim))

        if self.slicer is None:
            # same layout as on-disk
            self.spatial = self._spatial
            self.affine = self._affine
            self.shape = self._shape
            self.slicer = expand_index([Ellipsis], self._shape)

        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_if_mine()

    def __del__(self):
        self.close_if_mine()

    def close_if_mine(self):
        return self

    def close(self):
        return self

    def __str__(self):
        return '{}(shape={}, dtype={})'.format(
            type(self).__name__, self.shape, self.dtype)

    __repr__ = __str__

    def __len__(self):
        if len(self.shape) > 0:
            return self.shape[0]
        else:
            return 0

    @classmethod
    def possible_extensions(cls):
        """List all possible extensions"""
        return tuple()

    def __getitem__(self, index):
        """Extract a sub-part of the array.

        Indices can only be slices, ellipses, integers or None.

        Parameters
        ----------
        index : tuple[slice or ellipsis or int or None]

        Returns
        -------
        subarray : type(self)
            MappedArray object, with the indexing operations and affine
            matrix relating to the new sub-array.

        """
        return self.slice(index)

    def slice(self, index, new_shape=None, _pre_expanded=False):
        """Extract a sub-part of the array.

        Indices can only be slices, ellipses, integers or None.

        Parameters
        ----------
        index : tuple[slice or ellipsis or int or None]

        Other Parameters
        ----------------
        new_shape : sequence[int], optional
            Output shape of the sliced object
        _pre_expanded : bool, default=False
            Set to True of `expand_index` has already been called on `index`

        Returns
        -------
        subarray : type(self)
            MappedArray object, with the indexing operations and affine
            matrix relating to the new sub-array.

        """
        index = expand_index(index, self.shape)
        new_shape = guess_shape(index, self.shape)
        if any(isinstance(idx, list) for idx in index) > 1:
            raise ValueError('List indices not currently supported '
                             '(otherwise we enter advanced indexing '
                             'territory and it becomes too complicated).')
        new = copy(self)
        new.shape = new_shape

        # compute new affine
        if self.affine is not None:
            spatial_shape = [sz for sz, msk in zip(self.shape, self.spatial)
                             if msk]
            spatial_index = [idx for idx in index if not is_newaxis(idx)]
            spatial_index = [idx for idx, msk in zip(spatial_index, self.spatial)
                             if msk]
            affine, _ = affine_sub(self.affine, spatial_shape, tuple(spatial_index))
        else:
            affine = None
        new.affine = affine

        # compute new slicer
        perm_shape = [self._shape[d] for d in self.permutation]
        new.slicer = compose_index(self.slicer, index, perm_shape)

        # compute new spatial mask
        spatial = []
        i = 0
        for idx in new.slicer:
            if is_newaxis(idx):
                spatial.append(False)
            else:
                # original axis
                if not is_droppedaxis(idx):
                    spatial.append(self._spatial[self.permutation[i]])
                i += 1
        new.spatial = tuple(spatial)

        return new

    def __setitem__(self, index, value):
        """Write scaled data to disk.

        Parameters
        ----------
        index : tuple
            Tuple of indices (see `__getitem__`)
        value : array or tensor
            Array-like with shape `self[index].shape`

        Returns
        -------
        self : type(self)

        """
        if isinstance(value, MappedArray):
            raise NotImplementedError
        else:
            self.__getitem__(index).set_fdata(value)
        return self

    def __call__(self, *args, **kwargs):
        """Get floating point data. See `fdata()`"""
        return self.fdata(*args, **kwargs)

    def __array__(self, dtype=None):
        """Convert to numpy array"""
        return self.fdata(dtype=dtype, numpy=True)

    def permute(self, dims):
        """Permute dimensions

        Parameters
        ----------
        dims : sequence[int]
            A permutation of `range(self.dim)`

        Returns
        -------
        permarray : type(self)
            MappedArray object, with the indexing operations and affine
            matrix reflecting the permutation.

        """
        dims = list(dims)
        if len(dims) != self.dim or len(dims) != len(set(dims)):
            raise ValueError('there should be as many (unique) dimensions '
                             'as the array\'s dimension. Got {} and {}.'
                             .format(len(set(dims)), self.dim))

        # permute tuples that relate to the current spatial dimensions
        # (that part is easy)
        shape = tuple(self.shape[d] for d in dims)
        spatial = tuple(self.spatial[d] for d in dims)

        # permute slicer
        # 1) permute non-dropped dimensions
        slicer_nodrop = list(filter(lambda x: not is_droppedaxis(x), self.slicer))
        slicer_nodrop = [slicer_nodrop[d] for d in dims]
        # 2) insert dropped dimensions
        slicer = []
        for idx in self.slicer:
            if is_droppedaxis(idx):
                slicer.append(idx)
            else:
                new_idx, *slicer_nodrop = slicer_nodrop
                slicer.append(new_idx)

        # permute permutation
        # 1) insert None where new axes and remove dropped axes
        old_perm = self.permutation
        new_perm = []
        drop_perm = []
        for idx in self.slicer:
            if is_newaxis(idx):
                new_perm.append(None)
                continue
            p, *old_perm = old_perm
            if not is_droppedaxis(idx):
                new_perm.append(p)
            else:
                drop_perm.append(p)
        # 2) permute
        new_perm = [new_perm[d] for d in dims]
        # 3) insert back dropped axes and remove new axes
        perm = []
        for idx in self.slicer:
            if is_droppedaxis(idx):
                p, *drop_perm = drop_perm
                perm.append(p)
                continue
            p, *new_perm = new_perm
            if not is_newaxis(p):
                perm.append(p)

        # permute affine
        # (it's a bit more complicated: we need to find the
        #  permutation of the *current* *spatial* dimensions)
        perm_spatial = [p for p in dims if self.spatial[p]]
        remap = list(sorted(perm_spatial))
        remap = [remap.index(p) for p in perm_spatial]
        affine, _ = affine_permute(self.affine, remap, self.shape)

        # create new object
        new = copy(self)
        new.shape = shape
        new.spatial = spatial
        new.permutation = tuple(perm)
        new.slicer = tuple(slicer)
        new.affine = affine
        return new

    def movedim(self, source, destination):

        dim = self.dim
        source = make_list(source)
        destination = make_list(destination)
        if len(destination) == 1:
            # we assume that the user wishes to keep moved dimensions
            # in the order they were provided
            destination = destination[0]
            if destination >= 0:
                destination = list(range(destination, destination + len(source)))
            else:
                destination = list(range(destination + 1 - len(source), destination + 1))
        if len(source) != len(destination):
            raise ValueError('Expected as many source as destination positions.')
        source = [dim + src if src < 0 else src for src in source]
        destination = [dim + dst if dst < 0 else dst for dst in destination]
        if len(set(source)) != len(source):
            raise ValueError(f'Expected source positions to be unique but got '
                             f'{source}')
        if len(set(destination)) != len(destination):
            raise ValueError(f'Expected destination positions to be unique but got '
                             f'{destination}')

        # compute permutation
        positions_in = list(range(dim))
        positions_out = [None] * dim
        for src, dst in zip(source, destination):
            positions_out[dst] = src
            positions_in[src] = None
        positions_in = filter(lambda x: x is not None, positions_in)
        for i, pos in enumerate(positions_out):
            if pos is None:
                positions_out[i], *positions_in = positions_in

        return self.permute(positions_out)

    def transpose(self, dim0, dim1):
        """Transpose two dimensions

        Parameters
        ----------
        dim0 : int
            First dimension
        dim1 : int
        Second dimension

        Returns
        -------
        permarray : type(self)
            MappedArray object, with the indexing operations and affine
            matrix reflecting the transposition.

        """
        permutation = list(range(self.dim))
        permutation[dim0] = dim1
        permutation[dim1] = dim0
        return self.permute(permutation)

    def raw_data(self):
        """Load the (raw) array in memory

        This function always returns a numpy array, since not all
        data types exist in torch.

        Returns
        -------
        dat : np.ndarray[self.dtype]

        """
        raise NotImplementedError(f'It looks like {type(self).__name__} '
                                  f'does not implement its data loader.')

    def data(self, dtype=None, device=None, casting='unsafe', rand=True,
             missing=None, cutoff=None, dim=None, numpy=False):
        """Load the array in memory

        Parameters
        ----------
        dtype : type or torch.dtype or np.dtype, optional
            Output data type. By default, keep the on-disk data type.
        device : torch.device, default='cpu'
            Output device.
        rand : bool, default=False
            If the on-disk dtype is not floating point, sample noise
            in the uncertainty interval.
        missing : float or sequence[float], optional
            Value(s) that correspond to missing values.
            No noise is added to them, and they are converted to NaNs
            (if possible) or zero (otherwise).
        cutoff : float or (float, float), default=(0, 1)
            Percentile cutoff. If only one value is provided, it is
            assumed to relate to the upper percentile.
        dim : int or list[int], optional
            Dimensions along which to compute percentiles.
            By default, they are computed on the flattened array.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe', 'rescale'}, default='unsafe'
            Controls what kind of data casting may occur:
                * 'no': the data types should not be cast at all.
                * 'equiv': only byte-order changes are allowed.
                * 'safe': only casts which can preserve values are allowed.
                * 'same_kind': only safe casts or casts within a kind,
                  like float64 to float32, are allowed.
                * 'unsafe': any data conversions may be done.
                * 'rescale': the input data is rescaled to match the dynamic
                  range of the output type. The minimum value in the data
                  is mapped to the minimum value of the data type and the
                  maximum value in the data is mapped to the maximum value
                  of the data type.
                * 'rescale_zero': the input data is rescaled to match the
                  dynamic range of the output type, but ensuring that
                  zero maps to zero.
                  > If the data is signed and cast to a signed datatype,
                    zero maps to zero, and the scaling is chosen so that
                    both the maximum and minimum value in the data fit
                    in the output dynamic range.
                  > If the data is signed and cast to an unsigned datatype,
                    negative values "wrap around" (as with an unsafe cast).
                  > If the data is unsigned and cast to a signed datatype,
                    values are kept positive (the negative range is unused).
        numpy : bool, default=False
            Return a numpy array rather than a torch tensor.

        Returns
        -------
        dat : tensor[dtype]

        """
        # --- sanity check before reading ---
        if not numpy:
            if dtype is not None:
                dtype = dtypes.dtype(dtype)
                if dtype.torch is None:
                    raise TypeError(
                        f'Data type {dtype} does not exist in PyTorch.'
                    )
            dtype = dtypes.dtype(dtype or self.dtype)
            dtype = dtypes.dtype(dtype.torch_upcast)
        else:
            dtype = dtypes.dtype(dtype or self.dtype)

        # --- load raw data ---
        dat = self.raw_data()

        # --- move to tensor ---
        indtype = dtypes.dtype(self.dtype)
        if not numpy:
            tmpdtype = dtypes.dtype(indtype.torch_upcast)
            dat = dat.astype(tmpdtype.numpy)
            dat = torch.as_tensor(dat, dtype=indtype.torch_upcast, device=device)

        # --- mask of missing values ---
        if missing is not None:
            missing = volutils.missing(dat, missing)
            present = ~missing
        else:
            present = (Ellipsis,)

        # --- cutoff ---
        if cutoff is not None:
            dat[present] = volutils.cutoff(dat[present], cutoff, dim)

        # --- cast + rescale ---
        rand = rand and not indtype.is_floating_point
        tmpdtype = dtypes.float64 if (rand and not dtype.is_floating_point) else dtype
        dat, scale = volutils.cast(dat, tmpdtype, casting, indtype=indtype,
                                   returns='dat+scale', mask=present)

        # --- random sample ---
        # uniform noise in the uncertainty interval
        if rand and not (scale == 1 and not dtype.is_floating_point):
            dat[present] = volutils.addnoise(dat[present], scale)

        # --- final cast ---
        dat = volutils.cast(dat, dtype, 'unsafe')

        # --- replace missing values ---
        if missing is not None:
            if dtype.is_floating_point:
                dat[missing] = float('nan')
            else:
                dat[missing] = 0
        return dat

    def fdata(self, dtype=None, device=None, rand=False, missing=None,
              cutoff=None, dim=None, numpy=False):
        """Load the scaled array in memory

        This function differs from `data` in several ways:
            * The output data type should be a floating point type.
            * If an affine scaling (slope, intercept) is defined in the
              file, it is applied to the data.
            * the default output data type is `torch.get_default_dtype()`.

        Parameters
        ----------
        dtype : dtype_like, optional
            Output data type. By default, use `torch.get_default_dtype()`.
            Should be a floating point type.
        device : torch.device, default='cpu'
            Output device.
        rand : bool, default=False
            If the on-disk dtype is not floating point, sample noise
            in the uncertainty interval.
        missing : float or sequence[float], optional
            Value(s) that correspond to missing values.
            No noise is added to them, and they are converted to NaNs.
        cutoff : float or (float, float), default=(0, 1)
            Percentile cutoff. If only one value is provided, it is
            assumed to relate to the upper percentile.
        dim : int or list[int], optional
            Dimensions along which to compute percentiles.
            By default, they are computed on the flattened array.
        numpy : bool, default=False
            Return a numpy array rather than a torch tensor.

        Returns
        -------
        dat : tensor[dtype]

        """
        # --- sanity check ---
        dtype = torch.get_default_dtype() if dtype is None else dtype
        info = dtypes.dtype(dtype)
        if not info.is_floating_point:
            raise TypeError('Output data type should be a floating point '
                            'type but got {}.'.format(dtype))

        # --- get unscaled data ---
        dat = self.data(dtype=dtype, device=device, rand=rand, missing=missing,
                        cutoff=cutoff, dim=dim, numpy=numpy)

        # --- scale ---
        if self.slope != 1:
            dat *= float(self.slope)
        if self.inter != 0:
            dat += float(self.inter)

        return dat

    def set_data(self, dat, casting='unsafe'):
        """Write (partial) data to disk.

        Parameters
        ----------
        dat : tensor
            Tensor to write on disk. It should have shape `self.shape`.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe', 'rescale'}, default='unsafe'
            Controls what kind of data casting may occur:
                * 'no': the data types should not be cast at all.
                * 'equiv': only byte-order changes are allowed.
                * 'safe': only casts which can preserve values are allowed.
                * 'same_kind': only safe casts or casts within a kind,
                  like float64 to float32, are allowed.
                * 'unsafe': any data conversions may be done.
                * 'rescale': the input data is rescaled to match the dynamic
                  range of the output type. The minimum value in the data
                  is mapped to the minimum value of the data type and the
                  maximum value in the data is mapped to the maximum value
                  of the data type.
                * 'rescale_zero': the input data is rescaled to match the
                  dynamic range of the output type, but ensuring that
                  zero maps to zero.
                  > If the data is signed and cast to a signed datatype,
                    zero maps to zero, and the scaling is chosen so that
                    both the maximum and minimum value in the data fit
                    in the output dynamic range.
                  > If the data is signed and cast to an unsigned datatype,
                    negative values "wrap around" (as with an unsafe cast).
                  > If the data is unsigned and cast to a signed datatype,
                    values are kept positive (the negative range is unused).

        Returns
        -------
        self : type(self)

        """
        raise self.FailedWriteError("Method not implemented in class {}."
                                    .format(type(self).__name__))

    def set_fdata(self, dat):
        """Write (partial) scaled data to disk.

        Parameters
        ----------
        dat : tensor
            Tensor to write on disk. It should have shape `self.shape`
            and a floating point data type.

        Returns
        -------
        self : type(self)

        """
        # --- sanity check ---
        info = dtypes.dtype(dat.dtype)
        if not info.is_floating_point:
            raise TypeError('Input data type should be a floating point '
                            'type but got {}.'.format(dat.dtype))
        if dat.shape != self.shape:
            raise TypeError('Expected input shape {} but got {}.'
                            .format(self.shape, dat.shape))

        # --- detach ---
        if torch.is_tensor(dat):
            dat = dat.detach()

        # --- unscale ---
        if self.inter != 0 or self.slope != 1:
            dat = dat.clone() if torch.is_tensor(dat) else dat.copy()
        if self.inter != 0:
            dat -= float(self.inter)
        if self.slope != 1:
            dat /= float(self.slope)

        # --- set unscaled data ---
        self.set_data(dat)

        return self

    def metadata(self, keys=None):
        """Read metadata

        .. note:: The values returned by this function always relate to
                  the full volume, even if we're inside a view. That is,
                  we always return the affine of the original volume.
                  To get an affine matrix that relates to the view,
                  use `self.affine`.

        Parameters
        ----------
        keys : sequence[str], optional
            List of metadata to load. They can either be one of the
            generic metadata keys define in `io.metadata`, or a
            format-specific metadata key.
            By default, all generic keys that are found in the file
            are returned.

        Returns
        -------
        metadata : dict
            A dictionary of metadata

        """
        return dict()

    def set_metadata(self, **meta):
        """Write metadata

        Parameters
        ----------
        meta : dict, optional
            Dictionary of metadata.
            Fields that are absent from the dictionary or that have
            value `None` are kept untouched.

        Returns
        -------
        self : type(self)

        """
        raise NotImplementedError("Method not implemented in class {}."
                                  .format(type(self).__name__))

    @classmethod
    def save_new(cls, dat, file_like, like=None, casting='unsafe', **metadata):
        """Write an array to disk.

        This function makes educated choices for the file format and
        its metadata based on the file extension, the data type and the
        other options provided.

        Parameters
        ----------
        dat : tensor or array or MappedArray
            Data to write
        file_like : str or file object
            Path to file or file object (with methods `seek`, `read`).
            If the extension is known, it gets priority over `like` when
            choosing the output format.
        like : file or MappedArray
            An array on-disk that should be used as a template for the new
            file. Its metadata/layout/etc will be mimicked as much as possible.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe', 'rescale'}, default='unsafe'
            Controls what kind of data casting may occur.
            See `MappedArray.set_data`
        metadata : dict
            Metadata to store on disk. Values provided there will have
            priority over `like`.

        Returns
        -------
        dat : array or tensor
            The array loaded in memory
        attributes : dict, if attributes is not None
            Dictionary of attributes loaded as well

        """
        raise cls.FailedWriteError("Method not implemented in class {}."
                                   .format(cls.__name__))

    @classmethod
    def savef_new(cls, dat, file_like, like=None, **metadata):
        """Write a scaled array to disk.

        This function makes educated choices for the file format and
        its metadata based on the file extension, the data type and the
        other options provided.

        The input data type must be a floating point type.

        Parameters
        ----------
        dat : tensor or array or MappedArray
            Data to write
        file_like : str or file object
            Path to file or file object (with methods `seek`, `read`).
            If the extension is known, it gets priority over `like` when
            choosing the output format.
        like : file or MappedArray
            An array on-disk that should be used as a template for the new
            file. Its metadata/layout/etc will be mimicked as much as possible.
        metadata : dict
            Metadata to store on disk. Values provided there will have
            priority over `like`.

        Returns
        -------
        dat : array or tensor
            The array loaded in memory
        attributes : dict, if attributes is not None
            Dictionary of attributes loaded as well

        """
        raise cls.FailedWriteError("Method not implemented in class {}."
                                   .format(cls.__name__))

    def unsqueeze(self, dim, ndim=1):
        """Add a dimension of size 1 in position `dim`.

        Parameters
        ----------
        dim : int
            The dimension is added to the right of `dim` if `dim < 0`
            else it is added to the left of `dim`.

        Returns
        -------
        MappedArray

        """
        index = [slice(None)] * self.dim
        if dim < 0:
            dim = self.dim + dim + 1
        index = index[:dim] + ([None] * ndim) + index[dim:]
        return self[tuple(index)]

    def squeeze(self, dim):
        """Remove all dimensions of size 1.

        Parameters
        ----------
        dim : int or sequence[int], optional
            If provided, only this dimension is squeezed. It *must* be a
            dimension of size 1.

        Returns
        -------
        MappedArray

        """
        if dim is None:
            dim = [d for d in range(self.dim) if self.shape[d] == 1]
        dim = make_list(dim)
        ndim = len(self.shape)
        dim = [ndim + d if d < 0 else d for d in dim]
        if any(self.shape[d] != 1 for d in dim):
            raise ValueError('Impossible to squeeze non-singleton dimensions.')
        index = [slice(None) if d not in dim else 0 for d in range(self.dim)]
        return self[tuple(index)]

    def unbind(self, dim=0, keepdim=False):
        """Extract all arrays along dimension `dim` and drop that dimension.

        Parameters
        ----------
        dim : int, default=0
            Dimension along which to unstack.
        keepdim : bool, default=False
            Do not drop the unstacked dimension.

        Returns
        -------
        list[MappedArray]

        """
        index = [slice(None)] * self.dim
        if keepdim:
            index = index[:dim+1] + [None] + index[dim+1:]
        out = []
        for i in range(self.shape[dim]):
            index[dim] = i
            out.append(self[tuple(index)])
        return out

    def chunk(self, chunks, dim=0):
        """Split the array into smaller arrays of size `chunk` along `dim`.

        Parameters
        ----------
        chunks : int
            Number of chunks.
        dim : int, default=0
            Dimensions along which to split.

        Returns
        -------
        list[MappedArray]

        """
        index = [slice(None)] * self.dim
        out = []
        for i in range(self.shape[dim]):
            index[dim] = slice(i*chunks, (i+1)*chunks)
            out.append(self[tuple(index)])
        return out

    def split(self, chunks, dim=0):
        """Split the array into smaller arrays along `dim`.

        Parameters
        ----------
        chunks : int or list[int]
            If `int`: Number of chunks (see `self.chunk`)
            Else: Size of each chunk. Must sum to `self.shape[dim]`.
        dim : int, default=0
            Dimensions along which to split.

        Returns
        -------
        list[MappedArray]

        """
        if isinstance(chunks, int):
            return self.chunk(chunks, dim)
        chunks = make_list(chunks)
        if sum(chunks) != self.shape[dim]:
            raise ValueError('Chunks must cover the full dimension. '
                             'Got {} and {}.'
                             .format(sum(chunks), self.shape[dim]))
        index = [slice(None)] * self.dim
        previous_chunks = 0
        out = []
        for chunk in chunks:
            index[dim] = slice(previous_chunks, previous_chunks+chunk)
            out.append(self[tuple(index)])
            previous_chunks += chunk
        return out

    def flip(self, dim):
        dim = make_list(dim)
        slicer = [slice(None)] * self.dim
        for d in dim:
            slicer[d] = slice(None, None, -1)
        return self[tuple(slicer)]

    def channel_first(self, atleast=0):
        """Permute the dimensions such that all spatial axes are on the right.

        Parameters
        ----------
        atleast : int, default=0
            Make sure that at least this number of non-spatial dimensions
            exist (new axes are inserted accordingly).

        Returns
        -------
        MappedArray

        """
        # 1) move spatial dimensions to the right
        perm = []
        spatial = []
        for d, is_spatial in enumerate(self.spatial):
            if is_spatial:
                spatial.append(d)
            else:
                perm.append(d)
        nb_channels = len(perm)
        perm = perm + spatial
        new = self.permute(perm)
        # 2) add channel axes
        add_channels = max(0, atleast - nb_channels)
        if add_channels:
            index = [slice(None)] * nb_channels \
                  + [None] * add_channels \
                  + [Ellipsis]
            new = new.slice(tuple(index))
        return new

    def channel_last(self, atleast=0):
        """Permute the dimensions such that all spatial axes are on the left.

        Parameters
        ----------
        atleast : int, default=0
            Make sure that at least this number of non-spatial dimensions
            exist (new axes are inserted accordingly).

        Returns
        -------
        MappedArray

        """
        # 1) move spatial dimensions to the right
        perm = []
        spatial = []
        for d, is_spatial in enumerate(self.spatial):
            if is_spatial:
                spatial.append(d)
            else:
                perm.append(d)
        nb_channels = len(perm)
        perm = spatial + perm
        new = self.permute(perm)
        # 2) add channel axes
        add_channels = max(0, atleast - nb_channels)
        if add_channels:
            index = [Ellipsis] + [None] * add_channels
            new = new.slice(tuple(index))
        return new


class CatArray(MappedArray):
    """A concatenation of mapped arrays.

    This is largely inspired by virtual concatenation of file_array in
    SPM: https://github.com/spm/spm12/blob/master/@file_array/cat.m

    """

    _arrays: tuple = []
    _dim_cat: int = None

    # defer attributes
    fname = property(lambda self: tuple(a.fname for a in self._arrays))
    fileobj = property(lambda self: tuple(a.fileobj for a in self._arrays))
    is_compressed = property(lambda self: tuple(a.is_compressed for a in self._arrays))
    dtype = property(lambda self: tuple(a.dtype for a in self._arrays))
    slope = property(lambda self: tuple(a.slope for a in self._arrays))
    inter = property(lambda self: tuple(a.inter for a in self._arrays))
    _shape = property(lambda self: tuple(a._shape for a in self._arrays))
    _dim = property(lambda self: tuple(a._dim for a in self._arrays))
    affine = property(lambda self: tuple(a.affine for a in self._arrays))
    _affine = property(lambda self: tuple(a._affine for a in self._arrays))
    spatial = property(lambda self: tuple(a.spatial for a in self._arrays))
    _spatial = property(lambda self: tuple(a._spatial for a in self._arrays))
    slicer = property(lambda self: tuple(a.slicer for a in self._arrays))
    permutation = property(lambda self: tuple(a.permutation for a in self._arrays))
    voxel_size = property(lambda self: tuple(a.voxel_size for a in self._arrays))

    def __init__(self, arrays, dim=0):
        """

        Parameters
        ----------
        arrays : sequence[MappedArray]
            Arrays to concatenate. Their shapes should be identical
            except along dimension `dim`.
        dim : int, default=0
            Dimension along white to concatenate the arrays
        """
        super().__init__()

        arrays = list(arrays)
        dim = dim or 0
        self._dim_cat = dim

        # sanity checks
        shapes = []
        for i, array in enumerate(arrays):
            if not isinstance(array, MappedArray):
                raise TypeError('Input arrays should be `MappedArray` '
                                'instances. Got {}.',format(type(array)))
            shape = list(array.shape)
            del shape[dim]
            shapes.append(shape)
        shape0, *shapes = shapes
        if not all(shape == shape0 for shape in shapes):
            raise ValueError('Shapes of all concatenated arrays should '
                             'be equal except in the concatenation dimension.')

        # compute output shape
        shape = list(arrays[0].shape)
        dims = [array.shape[dim] for array in arrays]
        shape[dim] = sum(dims)
        self.shape = tuple(shape)

        # concatenate
        self._arrays = tuple(arrays)

    def __str__(self):
        dtype_str = tuple(str(dt) for dt in self.dtype)
        dtype_str = '(' + ', '.join(dtype_str) + ')'
        return '{}(shape={}, dtype={})'.format(
            type(self).__name__, self.shape, dtype_str)

    __repr__ = __str__

    def slice(self, index, new_shape=None):
        # overload slicer -> slice individual arrays
        index = expand_index(index, self.shape)
        new_shape = guess_shape(index, self.shape)
        assert len(index) > 0, "index should never be empty here"
        if any(isinstance(idx, list) for idx in index) > 1:
            raise ValueError('List indices not currently supported '
                             '(otherwise we enter advanced indexing '
                             'territory and it becomes too complicated).')
        index = list(index)
        shape_cat = self.shape[self._dim_cat]

        # find out which index corresponds to the concatenated dimension
        # + compute the concatenated dimension in the output array
        new_dim_cat = self._dim_cat
        nb_old_dim = -1
        for map_dim_cat, idx in enumerate(index):
            if is_newaxis(idx):
                # an axis was added: dim_cat moves to the right
                new_dim_cat = new_dim_cat + 1
            elif is_droppedaxis(idx):
                # an axis was dropped: dim_cat moves to the left
                new_dim_cat = new_dim_cat - 1
                nb_old_dim += 1
            else:
                nb_old_dim += 1
            if nb_old_dim >= self._dim_cat:
                # found the concatenated dimension
                break
        index_cat = index[map_dim_cat]
        index_cat = neg2pos(index_cat, shape_cat)  # /!\ do not call it again

        if is_droppedaxis(index_cat):
            # if the concatenated dimension is dropped, return the
            # corresponding array (sliced)
            if index_cat < 0 or index_cat >= shape_cat:
                raise IndexError('Index {} out of bounds [0, {}]'
                                 .format(index_cat, shape_cat))
            nb_pre = 0
            for i in range(len(self._arrays)):
                if nb_pre < index_cat:
                    # we haven't found the volume yet
                    nb_pre += self._arrays[i].shape[self._dim_cat]
                    continue
                break
            if nb_pre > index_cat:
                i = i - 1
                nb_pre -= self._arrays[i].shape[self._dim_cat]
            index_cat = index_cat - nb_pre
            index[map_dim_cat] = index_cat
            return self._arrays[i].slice(tuple(index), new_shape)

        # else, we may have to drop some volumes and slice the others
        assert is_sliceaxis(index_cat), "This should not happen"
        arrays = self._arrays

        step = index_cat.step or 1
        if step < 0:
            # if negative step:
            # 1) invert everything
            invert_index = [slice(None)] * self.dim
            invert_index[self._dim_cat] = slice(None, None, -1)
            arrays = [array[tuple(invert_index)] for array in arrays]
            # 2) update index_cat
            index_cat = invert_slice(index_cat, shape_cat, neg2pos=False)

        # compute navigator
        # (step is positive)
        start, step, nb_elem_total = slice_navigator(index_cat, shape_cat, do_neg2pos=False)

        nb_pre = 0              # nb of slices on the left of the cursor
        kept_arrays = []        # arrays at least partly in bounds
        starts = []             # start in each kept array
        stops = []              # stop in each kept array
        size_since_start = 0    # nb of in-bounds slices left of the cursor
        while len(arrays) > 0:
            # pop array
            array, *arrays = arrays
            size_cat = array.shape[self._dim_cat]
            if nb_pre + size_cat < start:
                # discarded volumes at the beginning
                nb_pre += size_cat
                continue
            if nb_pre < start:
                # first volume
                kept_arrays.append(array)
                starts.append(start - nb_pre)
            elif index_cat.stop is None or nb_pre < index_cat.stop:
                # other kept volume
                kept_arrays.append(array)
                skip = size_since_start - (size_since_start // step) * step
                starts.append(skip)
            # compute stopping point
            nb_elem_prev = size_since_start // step
            nb_elem_remaining = nb_elem_total - nb_elem_prev
            nb_elem_this_volume = (size_cat - starts[-1]) // step
            if nb_elem_remaining <= nb_elem_this_volume:
                # last volume
                stops.append(nb_elem_remaining)
                break
            # read as much as possible
            size_since_start += size_cat
            nb_pre += size_cat
            stops.append(None)
            continue

        # slice kept arrays
        arrays = []
        for array, start, stop in zip(kept_arrays, starts, stops):
            index[map_dim_cat] = slice(start, stop, step)
            arrays.append(array[tuple(index)])

        # create new CatArray
        new = copy(self)
        new._arrays = arrays
        new._dim_cat = new_dim_cat
        new.shape = new_shape
        return new

    def permute(self, dims):
        # overload permutation -> permute individual arrays
        new = copy(self)
        new._arrays = [array.permute(dims) for array in new._arrays]
        iperm = invert_permutation(dims)
        new._dim_cat = iperm[new._dim_cat]
        new.shape = tuple(self.shape[d] for d in dims)
        return new

    def raw_data(self, *args, **kwargs):
        # read individual arrays and concatenate them
        # TODO: it would be more efficient to preallocate the whole
        #   array and pass the appropriate buffer to each reader but
        #   (1) we don't have the option to provide a buffer yet
        #   (2) everything's already quite inefficient
        dats = [array.raw_data(*args, **kwargs) for array in self._arrays]
        return volutils.cat(dats, dim=self._dim_cat)

    def data(self, *args, **kwargs):
        # read individual arrays and concatenate them
        # TODO: it would be more efficient to preallocate the whole
        #   array and pass the appropriate buffer to each reader but
        #   (1) we don't have the option to provide a buffer yet
        #   (2) everything's already quite inefficient
        dats = [array.data(*args, **kwargs) for array in self._arrays]
        return volutils.cat(dats, dim=self._dim_cat)

    def fdata(self, *args,  **kwargs):
        # read individual arrays and concatenate them
        # TODO: it would be more efficient to preallocate the whole
        #   array and pass the appropriate buffer to each reader but
        #   (1) we don't have the option to provide a buffer yet
        #   (2) everything's already quite inefficient
        dats = [array.fdata(*args, **kwargs) for array in self._arrays]
        return volutils.cat(dats, dim=self._dim_cat)

    def set_data(self, dat, *args, **kwargs):
        # slice the input data and write it into each array
        size_prev = 0
        index = [None] * self.dim
        for array in self._arrays:
            size_cat = array.shape[self._dim_cat]
            index[self._dim_cat] = slice(size_prev, size_prev + size_cat)
            array._set_data(dat[tuple(index)], *args, **kwargs)

    def set_fdata(self, dat, *args, **kwargs):
        # slice the input data and write it into each array
        size_prev = 0
        index = [None] * self.dim
        for array in self._arrays:
            size_cat = array.shape[self._dim_cat]
            index[self._dim_cat] = slice(size_prev, size_prev + size_cat)
            array._set_fdata(dat[tuple(index)], *args, **kwargs)

    def metadata(self, *args, **kwargs):
        return tuple(array.metadata(*args, **kwargs) for array in self._arrays)

    def set_metadata(self, **meta):
        raise NotImplementedError('Cannot write metadata into concatenated '
                                  'array')


def cat(arrays, dim=0):
    """Concatenate mapped arrays along a dimension.

    Parameters
    ----------
        arrays : sequence[MappedArray]
            Arrays to concatenate. Their shapes should be identical
            except along dimension `dim`.
        dim : int, default=0
            Dimension along white to concatenate the arrays

    Returns
    -------
    CatArray
        A symbolic concatenation of all input arrays.
        Its shape along dimension `dim` is the sum of all input shapes
        along dimension `dim`.
    """
    return CatArray(arrays, dim)


def stack(arrays, dim=0):
    """Stack mapped arrays along a dimension.

    Parameters
    ----------
        arrays : sequence[MappedArray]
            Arrays to concatenate. Their shapes should be identical
            except along dimension `dim`.
        dim : int, default=0
            Dimension along white to concatenate the arrays

    Returns
    -------
    CatArray
        A symbolic stack of all input arrays.

    """
    arrays = [array.unsqueeze(dim=dim) for array in arrays]
    return cat(arrays, dim=dim)
