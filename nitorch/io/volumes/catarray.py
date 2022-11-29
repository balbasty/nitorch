from ..utils import indexing as idxutils, volumeutils as volutils
from .mappedarray import MappedArray


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


def _catproperty(key):
    def _get(self):
        return tuple(getattr(a, key) for a in self._arrays)
    return property(_get)


class CatArray(MappedArray):
    """A concatenation of mapped arrays.

    This is largely inspired by virtual concatenation of file_array in
    SPM: https://github.com/spm/spm12/blob/master/@file_array/cat.m

    """

    _arrays: tuple = []             # List of concatenated arrays
    _dim_cat: int = None            # Dimension along which to concat

    # defer attributes
    fname = _catproperty('fname')
    fileobj = _catproperty('fileobj')
    is_compressed = _catproperty('is_compressed')
    dtype = _catproperty('dtype')
    slope = _catproperty('slope')
    inter = _catproperty('inter')
    shaped = _catproperty('shaped')
    affine = _catproperty('affine')
    spatial = _catproperty('spatial')
    index = _catproperty('index')
    permutation = _catproperty('permutation')
    voxel_size = _catproperty('voxel_size')

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

        # sanity checks
        shapes = []
        for i, array in enumerate(arrays):
            if not isinstance(array, MappedArray):
                raise TypeError(f'Input arrays should be `MappedArray` '
                                f'instances. Got {type(array)}.')
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
        shape = tuple(shape)

        # concatenate
        super().__init__(shape=shape)
        self._dim_cat = dim
        self._arrays = tuple(arrays)

    def __str__(self):
        dtype_str = tuple(str(dt) for dt in self.dtype)
        dtype_str = '(' + ', '.join(dtype_str) + ')'
        return '{}(shape={}, dtype={})'.format(
            type(self).__name__, self.shape, dtype_str)

    def slice_(self, index, _outofplace=False):
        if not isinstance(index, tuple):
            index = (index,)

        # overload slicer -> slice individual arrays
        index = idxutils.expand_index(index, self.shape)
        new_shape = idxutils.guess_shape(index, self.shape)
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
            if idxutils.is_newaxis(idx):
                # an axis was added: dim_cat moves to the right
                new_dim_cat = new_dim_cat + 1
            elif idxutils.is_droppedaxis(idx):
                # an axis was dropped: dim_cat moves to the left
                new_dim_cat = new_dim_cat - 1
                nb_old_dim += 1
            else:
                nb_old_dim += 1
            if nb_old_dim >= self._dim_cat:
                # found the concatenated dimension
                break
        index_cat = index[map_dim_cat]
        index_cat = idxutils.neg2pos(index_cat, shape_cat)
        if isinstance(index_cat, idxutils.oob_int):
            raise IndexError(f'Index {index[map_dim_cat]} out of bounds '
                             f'[0, {shape_cat-1}]')

        if idxutils.is_droppedaxis(index_cat):
            # if the concatenated dimension is dropped, return the
            # corresponding array (sliced)
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
            array = self._arrays[i].slice(tuple(index))
            if _outofplace:
                return array
            arrays = [array]
            new_dim_cat = 0

        else:
            # else, we may have to drop some volumes and slice the others
            assert idxutils.is_sliceaxis(index_cat), "This should not happen"
            arrays = self._arrays

            step = index_cat.step or 1
            if step < 0:
                # if negative step:
                # 1) invert everything
                invert_index = [slice(None)] * self.dim
                invert_index[self._dim_cat] = slice(None, None, -1)
                arrays = [array[tuple(invert_index)] for array in arrays]
                # 2) update index_cat
                index_cat = idxutils.invert_slice(index_cat, shape_cat)

            # compute navigator
            # (step is positive)
            start, step, nb_elem_total = idxutils.slice_navigator(index_cat, shape_cat)

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
                arrays.append(array.slice(tuple(index)))

        # create new CatArray
        self._arrays = tuple(arrays)
        self._dim_cat = new_dim_cat
        self.shape = new_shape
        return self

    def permute_(self, dims):
        # overload permutation -> permute individual arrays
        self._arrays = tuple(array.permute(dims) for array in self._arrays)
        iperm = idxutils.invert_permutation(dims)
        self._dim_cat = iperm[self._dim_cat]
        self.shape = tuple(self.shape[d] for d in dims)
        return self

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

