from abc import ABC, abstractmethod
import torch
from copy import copy
from .indexing import expand_index, compose_index
from ..spatial import affine_sub, affine_permute
from ..spatial import voxel_size as affvx


class MappedArray(ABC):
    """Base class for mapped arrays.

    Mapped arrays are usually stored on-disk, along with (diverse) metadata.

    """

    fname: str = None             # filename (can be None if in-memory proxy)
    fileobj = None                # file-like object (`write`, `seek`, etc)
    is_compressed: bool = None    # is compressed
    metadata: dict = dict()       # human-readable metadata (function?)
    dtype: torch.dtype = None     # on-disk data type

    affine = None                 # voxel-to-world
    _affine = None                # original voxel-to-world
    spatial: tuple = None         # mask of the spatial dimensions
    _spatial: tuple = None        # original spatial mask
    shape: tuple = None           # array shape (spatial and others)
    _shape: tuple = None          # original shape
    slicer: tuple = None          # indexing into the parent
    perm: tuple = None            # permutation of the parent's dimensions

    dim = property(lambda self: len(self.shape))
    _dim = property(lambda self: len(self._shape))
    batch = property(lambda self: [d for i, d in enumerate(self.shape)
                                   if i not in self.spatial])
    voxel_size = property(lambda self: affvx(self.affine))

    def __init__(self, **kwargs):
        self._init(**kwargs)

    def _init(self, **kwargs):
        for key, val in kwargs:
            setattr(self, key, val)

        if self.perm is None:
            self.perm = tuple(range(self._dim))

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

        # update permutation (add/drop axes)
        perm = []
        i = 0
        for idx in index:
            if idx is None or (isinstance(idx, int) and idx < 0):
                # new axis
                perm.append(None)
                continue
            elif isinstance(idx, int):
                # dropped axis
                i += 1
            else:
                # kept axis
                perm.append(self.perm[i])
                i += 1
        new.perm = tuple(perm)

        # compute new slicer
        if self.slicer is None:
            new.slicer = index
        else:
            new.slicer = compose_index(self.slicer, index)

        # compute new mask of the spatial dimensions
        new.spatial = tuple(False if p is None else self._spatial[p]
                            for p in new.perm)
        new.affine = affine
        return new

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

        # permutations
        perm = tuple(self.perm[d] for d in dims)
        shape = tuple(self.shape[d] for d in dims)
        slicer = tuple(self.slicer[d] for d in dims)
        spatial = tuple(self.spatial[d] for d in dims)

        # permute affine
        # (it's a bit more complicated: we need to find the
        #  permutation of the *current* spatial dimensions)
        perm_spatial = [p for p in dims if self.spatial[p]]
        perm_spatial = sorted(range(len(perm_spatial)),
                              key=lambda k: perm_spatial[k])
        affine, _ = affine_permute(self.affine, self.shape, perm_spatial)

        # create new object
        new = copy(self)
        new.perm = perm
        new.shape = shape
        new.slicer = slicer
        new.spatial = spatial
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
        perm = list(range(self.dim))
        perm[dim0] = dim1
        perm[dim1] = dim0
        return self.permute(perm)

    @abstractmethod
    def data(self, dtype=None, device=None, casting='unsafe', numpy=False):
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

    @abstractmethod
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
        pass

