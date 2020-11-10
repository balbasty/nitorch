from abc import ABC, abstractmethod
import torch
from copy import copy
from .indexing import expand_index, compose_index, \
                      is_droppedaxis, is_newaxis, is_sliceaxis
from ..spatial import affine_sub, affine_permute
from ..spatial import voxel_size as affvx
from . import dtype as cast_dtype


class MappedArray(ABC):
    """Base class for mapped arrays.

    Mapped arrays are usually stored on-disk, along with (diverse) metadata.

    They can be symbolically sliced, allowing for partial reading and
    (sometimes) writing of data from/to disk.
    Chaining of symbolic slicing operations is implemented in this base
    class. The actual partial io must be implemented by the child class.
    """

    fname: str = None             # filename (can be None if in-memory proxy)
    fileobj = None                # file-like object (`write`, `seek`, etc)
    is_compressed: bool = None    # is compressed
    dtype: torch.dtype = None     # on-disk data type
    slope: float = None           # intensity slope
    inter: float = None           # intensity shift

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

        return self

    def __getitem__(self, index):
        """Extract a sub-part of the array.

        Indices can only be slices, ellipses, lists or integers.
        Indices *into spatial dimensions* cannot be lists.

        Parameters
        ----------
        index : tuple[slice or ellipsis or int]

        Returns
        -------
        subarray : type(self)
            MappedArray object, with the indexing operations and affine
            matrix relating to the new sub-array.

        """
        new = copy(self)
        index, new.shape = expand_index(index, self.shape)
        if any(isinstance(idx, list) for idx in index) > 1:
            raise ValueError('List indices not currently supported '
                             '(otherwise we enter advanced indexing '
                             'territory and it becomes too complicated).')

        # compute new affine
        if self.affine is not None:
            spatial_shape = [sz for sz, msk in zip(self.shape, self.spatial)
                             if msk]
            spatial_index = [idx for idx in index if isinstance(idx, slice)
                             or (isinstance(idx, int) and idx >= 0)]
            spatial_index = [idx for idx, msk in zip(spatial_index, self.spatial)
                             if msk]
            affine, _ = affine_sub(self.affine, spatial_shape, tuple(spatial_index))
        else:
            affine = None
        new.affine = affine

        # compute new slicer
        if self.slicer is None:
            new.slicer = index
        else:
            new.slicer = compose_index(self.slicer, index)

        # compute new spatial mask
        spatial = []
        i = 0
        for idx in index:
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
        return self.fdata(*args, **kwargs)

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

        # Permute tuples that relate to the current spatial dimensions
        shape = tuple(self.shape[d] for d in dims)
        spatial = tuple(self.spatial[d] for d in dims)

        # Permute tuples that relate to the slicer indices
        # (some of these slicers can drop dimensions, so their length
        #  can be greater than the current number of dimensions)
        slicer = []
        dim_map = []
        n_slicer = 0        # index into the slicer tuple
        n_dropped = 0       # number of dropped dimensions on the left
        for d in dims:
            if is_droppedaxis(self.slicer[n_slicer]):
                slicer.append(self.slicer[n_slicer])
                dim_map.append(self.permutation[n_slicer])
                n_dropped += 1
            else:
                slicer.append(self.slicer[d + n_dropped])
                dim_map.append(self.permutation[d + n_dropped])
            n_slicer += 1

        # permute affine
        # (it's a bit more complicated: we need to find the
        #  permutation of the *current* *spatial* dimensions)
        perm_spatial = [p for p in dims if self.spatial[p]]
        perm_spatial = sorted(range(len(perm_spatial)),
                              key=lambda k: perm_spatial[k])
        affine, _ = affine_permute(self.affine, self.shape, perm_spatial)

        # create new object
        new = copy(self)
        new.shape = shape
        new.spatial = spatial
        new.permutation = tuple(dim_map)
        new.slicer = tuple(slicer)
        new.affine = affine
        return new

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

    @abstractmethod
    def data(self, dtype=None, device=None, casting='unsafe', rand=True,
             cutoff=None, dim=None, numpy=False):
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
        pass

    def fdata(self, dtype=None, device=None, rand=False, cutoff=None,
              dim=None, numpy=False):
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

    @abstractmethod
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
        pass

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
        info = cast_dtype.info(dat.dtype)
        if not info['is_floating_point']:
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
            dat -= self.inter
        if self.slope != 1:
            dat /= self.slope

        # --- set unscaled data ---
        self.set_data(dat)

        return self

    @abstractmethod
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
        pass

    @abstractmethod
    def set_metadata(self, meta=None):
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
        pass
