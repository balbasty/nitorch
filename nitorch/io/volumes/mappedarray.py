import torch
from nitorch.core import dtypes
from ..utils import volumeutils as volutils
from ..utils.sliceable import SlicedSpatial
from ..mappedfile import MappedFile


class MappedArray(MappedFile, SlicedSpatial):
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

    raw_data(...) -> tensor     Load raw numpy data (without upcasting) to memory
    data(...) -> tensor         Load raw data to memory (optional upcasting)
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

    dtype: torch.dtype = None     # on-disk voxel data type
    slope: float = 1              # intensity slope
    inter: float = 0              # intensity shift

    def init(self, **kwargs):
        shaped = kwargs.pop('shaped', None)
        sliced_kwargs = dict(
            permutation=kwargs.pop('permutation', None),
            index=kwargs.pop('index', None),
            affine=kwargs.pop('affine', None),
            spatial=kwargs.pop('spatial', None),
            shape=kwargs.pop('shape', None),
        )
        SlicedSpatial.__init__(self, shaped, **sliced_kwargs)
        MappedFile.__init__(**kwargs)
        return self

    def __str__(self):
        return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype})'

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
            self.slice(index).set_fdata(value)
        return self

    def __call__(self, *args, **kwargs):
        """Get floating point data. See `fdata()`"""
        return self.fdata(*args, **kwargs)

    def __array__(self, dtype=None):
        """Convert to numpy array"""
        return self.fdata(dtype=dtype, numpy=True)

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
        dtype = self.dtype if dtype is None else dtype
        dtype = dtypes.dtype(dtype)
        if not numpy and dtype.torch is None:
            raise TypeError('Data type {} does not exist in PyTorch.'
                            .format(dtype))

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
