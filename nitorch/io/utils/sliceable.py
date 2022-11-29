import torch
from nitorch.spatial import affine_default, affine_sub, affine_permute
from nitorch.spatial import voxel_size as affvox
from nitorch.core.py import make_list
from . import indexing
from ..optionals import numpy as np
from copy import copy, deepcopy
from typing import List, Tuple, Optional, Sequence, Union


IndexLike = Union[int, slice, type(Ellipsis), type(None)]
IndexLike = Union[IndexLike, Tuple[IndexLike]]
ArrayLike = Union[torch.Tensor, getattr(np, 'ndarray', None), Sequence]


def maskfilter(mask, seq):
    for s, m in zip(seq, mask):
        if m:
            yield s


class Shaped:
    """An object that has a shape"""

    def __init__(self, shape):
        self.shape = tuple(shape)

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __repr__(self):
        return f'{type(self).__name__}({list(self.shape)})'

    def __str__(self):
        return repr(self)


class ShapedSpatial(Shaped):
    """An object that has a shape, whose some dimensions are spatial"""

    def __init__(self, shape: Sequence[int],
                 spatial: Union[int, Sequence[int], Sequence[bool]] = tuple(),
                 affine: ArrayLike = None):
        """
        Parameters
        ----------
        shape : sequence[int]
        spatial : int or sequence[int] or sequence[bool]
            Either:
                * the number of spatial dimensions (assumed to be last), or
                * the indices of the spatial dimensions, or
                * a mask of of the spatial dimensions.
        affine : (D+1, D+1) array-like
            An orientation matrix that maps voxel coordinates
            to world coordinates
        """
        super().__init__(shape)  # mixin

        if isinstance(spatial, int):
            spatial = range(-spatial, 0)
        spatial = list(spatial)
        spatial = spatial or ([False] * self.ndim)
        if isinstance(spatial[0], int):
            spatial_indices = spatial
            spatial = [False] * len(self.shape)
            for i in spatial_indices:
                spatial[i] = True
        if not all(map(lambda x: isinstance(x, bool), spatial)):
            raise TypeError('Expected a sequence of `bool`')
        self.spatial = tuple(spatial)

        if affine is None:
            affine = affine_default(self.spatial_shape)
        self.affine = torch.as_tensor(affine)

    @property
    def spatial_shape(self):
        return tuple(maskfilter(self.spatial, self.shape))

    @property
    def voxel_size(self):
        return affvox(self.affine)

    def __repr__(self):
        return f'{type(self).__name__}({list(self.shape)}, {list(self.spatial)})'


class Sliced:
    """An object that is a view into a (permuted) Shaped object"""

    def __init__(self, shaped: Union[Sequence[int], Shaped] = None, *,
                 index: Optional[IndexLike] = tuple(),
                 permutation: Optional[Sequence[int]] = tuple(), **kwargs):
        """

        Parameters
        ----------
        shaped : Shaped or sequence[int]
            Reference object, or shape
        index : (tuple of) {int, slice, Ellipsis, None}, optional
            Index into the reference object
        permutation : tuple[int]
            Permutation of the dimensions of the reference object.
            The permutation is applied *before* the indexing.
            (In other words, the indexing is applied to the *permuted* object)
        """
        if not isinstance(shaped, Shaped):
            kwargs.setdefault('shape', shaped)
            shaped = Shaped(**kwargs)
        self.shaped = shaped
        if not isinstance(index, tuple):
            index = (index,)
        self.index = indexing.expand_index(index or [Ellipsis], shaped.shape)
        self.permutation = tuple(permutation or range(len(shaped.shape)))
        self.shape = indexing.guess_shape(self.index, shaped.shape)

    def __repr__(self):
        s = f'{type(self).__name__}({list(self.shaped.shape)})'
        if self.permutation != tuple(range(self.shaped.ndim)):
            s += f'.permute({list(self.permutation)})'
        if self.index != (slice(None),) * self.shaped.ndim:
            s += f'[{repr_index(self.index)}]'
        return s

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __getitem__(self, item):
        """Alias for `slice`"""
        return self.slice(item)

    def __iter__(self):
        """Unbind along the first dimension"""
        for i in range(len(self)):
            yield self.slice(i)

    def slice(self, index: IndexLike):
        """Index into the object.

        Parameters
        ----------
        index : (tuple of) {int, slice, Ellipsis, None}

        Returns
        -------
        slice : Sliced
        """
        new = copy(self)
        return new.slice_(index)

    def slice_(self, index: IndexLike):
        """In-place version of `slice`"""
        if not isinstance(index, tuple):
            index = (index,)
        index = indexing.expand_index(index, self.shape)
        new_shape = indexing.guess_shape(self.index, self.shape)
        ref_shape = [self.shaped.shape[d] for d in self.permutation]
        new_index = indexing.compose_index(self.index, index, ref_shape)
        self.index = new_index
        self.shape = new_shape
        return self

    def permute(self, dims: Sequence[int]):
        """Permute dimensions

        Parameters
        ----------
        dims : sequence[int]

        Returns
        -------
        slice : Sliced

        """
        return copy(self).permute_(dims)

    def permute_(self, dims: Sequence[int]):
        """In-place version of `permute`"""
        dims = list(dims)
        if len(dims) != self.ndim or len(dims) != len(set(dims)):
            raise ValueError(f'there should be as many (unique) dimensions '
                             f'as the array\'s dimension. '
                             f'Got {len(set(dims))} and {self.ndim}.')

        # permute tuples that relate to the current spatial dimensions
        # (that part is easy)
        shape = tuple(self.shape[d] for d in dims)

        # permute slicer
        # 1) permute non-dropped dimensions
        isnot_droppedaxis = lambda x: not indexing.is_droppedaxis(x)
        index_nodrop = list(filter(isnot_droppedaxis, self.index))
        index_nodrop = [index_nodrop[d] for d in dims]
        # 2) insert dropped dimensions
        index = []
        for idx in self.index:
            if indexing.is_droppedaxis(idx):
                index.append(idx)
            else:
                new_idx, *index_nodrop = index_nodrop
                index.append(new_idx)

        # permute permutation
        # 1) insert None where new axes and remove dropped axes
        old_perm = self.permutation
        new_perm = []
        drop_perm = []
        for idx in self.index:
            if indexing.is_newaxis(idx):
                new_perm.append(None)
                continue
            p, *old_perm = old_perm
            if not indexing.is_droppedaxis(idx):
                new_perm.append(p)
            else:
                drop_perm.append(p)
        # 2) permute
        new_perm = [new_perm[d] for d in dims]
        # 3) insert back dropped axes and remove new axes
        perm = []
        for idx in self.index:
            if indexing.is_droppedaxis(idx):
                p, *drop_perm = drop_perm
                perm.append(p)
                continue
            p, *new_perm = new_perm
            if not indexing.is_newaxis(p):
                perm.append(p)

        self.index = index
        self.permutation = perm
        self.shape = shape
        return self

    def movedim(self, source, destination):
        """Move dimension(s) around

        Parameters
        ----------
        source : int or list[int]
            Indices of dimensions to move
        destination : int or list[int]
            Destination index of each source index

        Returns
        -------
        slice : Sliced

        """
        return copy(self).movedim_(source, destination)

    def movedim_(self, source: Union[int, Sequence[int]],
                 destination: Union[int, Sequence[int]]):
        """In-place version of `movedim`"""
        ndim = self.ndim
        source: List[int] = make_list(source)
        destination: List[int] = make_list(destination)
        if len(destination) == 1:
            # we assume that the user wishes to keep moved dimensions
            # in the order they were provided
            dest1 = destination[0]
            if dest1 >= 0:
                destination = list(range(dest1, dest1 + len(source)))
            else:
                destination = list(range(dest1 + 1 - len(source), dest1 + 1))
        if len(source) != len(destination):
            raise ValueError('Expected as many source as destination positions.')
        source = [ndim + src if src < 0 else src for src in source]
        destination = [ndim + dst if dst < 0 else dst for dst in destination]
        if len(set(source)) != len(source):
            raise ValueError(f'Expected source positions to be unique but got '
                             f'{source}')
        if len(set(destination)) != len(destination):
            raise ValueError(f'Expected destination positions to be unique but got '
                             f'{destination}')

        # compute permutation
        positions_in = list(range(ndim))
        positions_out = [None] * ndim
        for src, dst in zip(source, destination):
            positions_out[dst] = src
            positions_in[src] = None
        positions_in = filter(lambda x: x is not None, positions_in)
        for i, pos in enumerate(positions_out):
            if pos is None:
                positions_out[i], *positions_in = positions_in

        return self.permute_(positions_out)

    def transpose(self, dim0: int, dim1: int):
        """Transpose two dimensions

        Parameters
        ----------
        dim0 : int
            First dimension
        dim1 : int
            Second dimension

        Returns
        -------
        slice : Sliced

        """
        return copy(self).transpose_(dim0, dim1)

    def transpose_(self, dim0: int, dim1: int):
        """In-place version of `transpose`"""
        permutation = list(range(self.ndim))
        permutation[dim0] = dim1
        permutation[dim1] = dim0
        return self.permute_(permutation)

    def unsqueeze(self, dim: Union[int, List[int]], ndim: int = 1):
        """Add a dimension of size 1 in position `dim`.

        Parameters
        ----------
        dim : int or list[int]
            The dimension is added to the right of `dim` if `dim < 0`
            else it is added to the left of `dim`.
        ndim : int, default=1
            Number of dimensions to insert in each position.

        Returns
        -------
        Sliced

        """
        copy(self).unsqueeze_(dim, ndim)

    def unsqueeze_(self, dim: Union[int, List[int]], ndim: int = 1):
        """In-place version of `unsqueeze`"""
        index = [slice(None)] * self.ndim
        dim = sorted([ndim + d + 1 if d < 0 else d for d in make_list(dim)])
        count = 0
        for d in dim:
            d = d + count
            index = index[:d] + ([None] * ndim) + index[d:]
            count += ndim
        return self.slice_(tuple(index))

    def squeeze(self, dim: Optional[Union[int, List[int]]] = None):
        """Remove all dimensions of size 1.

        Parameters
        ----------
        dim : int or sequence[int], optional
            If provided, only this dimension is squeezed. It *must* be a
            dimension of size 1.

        Returns
        -------
        slice : Sliced

        """
        copy(self).squeeze_(dim)

    def squeeze_(self, dim: Optional[Union[int, List[int]]] = None):
        """In-place version of `squeeze`"""
        if dim is None:
            dim = [d for d in range(self.ndim) if self.shape[d] == 1]
        dim = make_list(dim)
        ndim = len(self.shape)
        dim = [ndim + d if d < 0 else d for d in dim]
        if any(self.shape[d] != 1 for d in dim):
            raise ValueError('Impossible to squeeze non-singleton dimensions.')
        index = [slice(None) if d not in dim else 0 for d in range(self.ndim)]
        return self[tuple(index)]

    def flip(self, dim: Optional[Union[int, List[int]]] = None):
        """Flip dimension(s)

        Parameters
        ----------
        dim : int or list[int], optional
            Dimension(s) to flip.
            If None, flip all dimensions

        Returns
        -------
        sliced : Sliced

        """
        return copy(self).flip_(dim)

    def flip_(self, dim: Optional[Union[int, List[int]]] = None):
        """In-place version of `flip`"""
        if dim is None:
            dim = list(range(self.ndim))
        dim = make_list(dim)
        index = [slice(None)] * self.ndim
        for d in dim:
            index[d] = slice(None, None, -1)
        return self.slice_(tuple(index))

    def unbind(self, dim: int = 0, keepdim: bool = False):
        """Extract all arrays along dimension `dim` and drop that dimension.

        Similar to `torch.unbind`

        Parameters
        ----------
        dim : int, default=0
            Dimension along which to unstack.
        keepdim : bool, default=False
            Do not drop the unstacked dimension.

        Returns
        -------
        list[Sliced]

        """
        index = [slice(None)] * self.ndim
        out = []
        for i in range(self.shape[dim]):
            index[dim] = slice(i, i+1) if keepdim else i
            out.append(self.slice(tuple(index)))
        return out

    def chunk(self, chunks: int, dim: int = 0):
        """Split the array into smaller arrays of size `chunk` along `dim`.

        Similar to `torch.chunk`

        If the length of the dimension cannot be divided by `chunks`,
        the chunk size is `length // chunks`, such that all chunks have
        the same size. The last, smaller, chunk is discarded.

        Parameters
        ----------
        chunks : int
            Number of chunks.
        dim : int, default=0
            Dimensions along which to split.

        Returns
        -------
        list[Sliced]

        """
        index = [slice(None)] * self.ndim
        out = []
        chunksize = self.shape[dim] // chunks
        for i in range(chunks):
            index[dim] = slice(i*chunksize, (i+1)*chunksize)
            out.append(self.slice(tuple(index)))
        return out

    def split(self, chunks: Union[int, Sequence[int]], dim: int = 0):
        """Split the array into smaller arrays along `dim`.

        Similar to `np.split`

        Parameters
        ----------
        chunks : int or sequence[int]
            If `int`: Number of chunks (see `self.chunk`)
            Else: Size of each chunk. Must sum to `self.shape[dim]`.
        dim : int, default=0
            Dimensions along which to split.

        Returns
        -------
        list[Sliced]

        """
        if isinstance(chunks, int):
            return self.chunk(chunks, dim)
        chunks = make_list(chunks)
        if sum(chunks) != self.shape[dim]:
            raise ValueError(f'Chunks must cover the full dimension. '
                             f'Got {sum(chunks)} and {self.shape[dim]}.')
        index = [slice(None)] * self.ndim
        previous_chunks = 0
        out = []
        for chunk in chunks:
            index[dim] = slice(previous_chunks, previous_chunks+chunk)
            out.append(self.slice(tuple(index)))
            previous_chunks += chunk
        return out


class SlicedSpatial(Sliced):
    """An object that is a view into a (permuted) ShapedSpatial object"""

    def __init__(self, shaped: Union[Sequence[int], ShapedSpatial] = None, *,
                 index: Optional[IndexLike] = tuple(),
                 permutation: Optional[Sequence[int]] = tuple(),
                 **kwargs):
        """

        Parameters
        ----------
        shaped : ShapedSpatial or sequence[int]
            Reference object or shape
        index : (tuple of) {int, slice, Ellipsis, None}, optional
            Index into the reference object
        permutation : tuple[int]
            Permutation of the dimensions of the reference object.
            The permutation is applied *before* the indexing.
            (In other words, the indexing is applied to the *permuted* object)
        """
        if not isinstance(shaped, ShapedSpatial):
            if isinstance(shaped, Shaped):
                shaped = shaped.shape
            kwargs.setdefault('shape', shaped)
            shaped = ShapedSpatial(**kwargs)
        super().__init__(shaped, index=index, permutation=permutation)

        if index or permutation:
            sliced = SlicedSpatial(shaped, **kwargs)
            sliced = sliced.permute_(permutation)
            sliced = sliced.slice_(index)
            spatial = sliced.spatial
            affine = sliced.affine
        else:
            spatial = shaped.spatial
            affine = shaped.affine
        self.affine = affine
        self.spatial = spatial

    def __repr__(self):
        s = f'{type(self).__name__}'
        s += f'({list(self.shaped.shape)}, {list(self.shaped.spatial)})'
        if self.permutation != tuple(range(self.shaped.ndim)):
            s += f'.permute({list(self.permutation)})'
        if self.index != (slice(None),) * self.shaped.ndim:
            s += f'[{repr_index(self.index)}]'
        return s

    @property
    def spatial_shape(self):
        return tuple(maskfilter(self.spatial, self.shape))

    @property
    def voxel_size(self):
        return affvox(self.affine)

    def slice_(self, index: IndexLike):
        if not isinstance(index, tuple):
            index = (index,)

        index = indexing.expand_index(index, self.shape)
        new_shape = indexing.guess_shape(index, self.shape)

        filtermask = lambda m, x: [x1 for x1, m1 in zip(x, m) if m1]
        isnot_newaxis = lambda x: not indexing.is_newaxis(x)

        # compute new affine
        spatial_shape = tuple(filtermask(self.spatial, self.shape))
        spatial_index = filter(isnot_newaxis, index)
        spatial_index = tuple(filtermask(self.spatial, spatial_index))
        new_affine, _ = affine_sub(self.affine, spatial_shape, spatial_index)

        # compute new slicer
        perm_shape = [self.shaped.shape[d] for d in self.permutation]
        new_index = indexing.compose_index(self.index, index, perm_shape)

        # compute new spatial mask
        spatial = []
        i = 0
        for idx in new_index:
            if indexing.is_newaxis(idx):
                spatial.append(False)
            else:
                # original axis
                if not indexing.is_droppedaxis(idx):
                    spatial.append(self.shaped.spatial[self.permutation[i]])
                i += 1
        new_spatial = tuple(spatial)

        self.index = tuple(new_index)
        self.shape = tuple(new_shape)
        self.affine = new_affine
        self.spatial = tuple(new_spatial)
        return self

    def permute_(self, dims: Sequence[int]):
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
        isnot_droppedaxis = lambda x: not indexing.is_droppedaxis(x)
        index_nodrop = list(filter(isnot_droppedaxis, self.index))
        index_nodrop = [index_nodrop[d] for d in dims]
        # 2) insert dropped dimensions
        index = []
        for idx in self.index:
            if indexing.is_droppedaxis(idx):
                index.append(idx)
            else:
                new_idx, *slicer_nodrop = index_nodrop
                index.append(new_idx)

        # permute permutation
        # 1) insert None where new axes and remove dropped axes
        old_perm = self.permutation
        new_perm = []
        drop_perm = []
        for idx in self.index:
            if indexing.is_newaxis(idx):
                new_perm.append(None)
                continue
            p, *old_perm = old_perm
            if not indexing.is_droppedaxis(idx):
                new_perm.append(p)
            else:
                drop_perm.append(p)
        # 2) permute
        new_perm = [new_perm[d] for d in dims]
        # 3) insert back dropped axes and remove new axes
        perm = []
        for idx in self.index:
            if indexing.is_droppedaxis(idx):
                p, *drop_perm = drop_perm
                perm.append(p)
                continue
            p, *new_perm = new_perm
            if not indexing.is_newaxis(p):
                perm.append(p)

        # permute affine
        # (it's a bit more complicated: we need to find the
        #  permutation of the *current* *spatial* dimensions)
        perm_spatial = [p for p in dims if self.spatial[p]]
        remap = list(sorted(perm_spatial))
        remap = [remap.index(p) for p in perm_spatial]
        affine, _ = affine_permute(self.affine, remap, self.shape)

        # update object
        self.shape = tuple(shape)
        self.index = tuple(index)
        self.affine = affine
        self.spatial = tuple(spatial)
        return self


def repr_index(index):
    reprindex = []
    for idx in index:
        if isinstance(idx, int):
            reprindex.append(str(idx))
        elif isinstance(idx, slice):
            reprslice = ''
            if idx.start is not None:
                reprslice += str(idx.start)
            reprslice += ':'
            if idx.stop is not None:
                reprslice += str(idx.stop)
            if idx.step not in (1, None):
                reprslice += ':' + str(idx.step)
            reprindex.append(reprslice)
        elif isinstance(idx, type(Ellipsis)):
            reprindex.append('...')
        elif idx is None:
            reprindex.append('None')
        else:
            raise TypeError(f'Unknown index type {type(idx)}')
    return ', '.join(reprindex)
