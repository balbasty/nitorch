from . import optionals
from functools import wraps
import torch
from ..core.optionals import numpy as np
from copy import copy


def expand_index(index, shape):
    """Expand a tuple of indices into an array of shape `shape`

    * Ellipses are replaced with slices
    * Start/Stop/Step values of all slices are computed
    * Implicit slices are appended on the right
    * The output shape is computed

    This function is related to `nibabel.fileslice.canonical_slicers`,
    but with a few differences:
    * Slices that span the whole length of a dimension are *not* replaced
      with `slice(None)`
    * Floating point indices are *not* accepted
    * Negative steps are *not* made positive
    * Lists of indices *are* accepted. However, we do not follow exactly
      the advanced indexing convention form numpy/pytorch. What we
      do is closer to the way Matlab handles indexing. As long as
      only one dimension is indexed using a list, both conventions are
      consistent.


    Parameters
    ----------
    index : tuple of index_like
        A tuple of indices (
    shape

    Returns
    -------

    """
    index = list(index)
    nb_dim = len(shape)

    def is_bool(elem):
        if torch.is_tensor(elem):
            return elem.dtype == torch.bool
        elif np and isinstance(elem, np.ndarray):
            return elem.dtype == np.bool
        elif isinstance(elem, bool):
            return True
        else:
            return all(isinstance(e, bool) for e in elem)

    def is_int(elem):
        if torch.is_tensor(elem):
            return elem.dtype in (torch.int32, torch.int64)
        elif np and isinstance(elem, np.ndarray):
            return elem.dtype in (np.int32, np.int64)
        elif isinstance(elem, int):
            return True
        else:
            return all(isinstance(e, int) for e in elem)

    # compute the number of input dimension that correspond to each index
    # an individual index can be a slice, int, ellipsis, array_like[int]
    # or array_like[bool]
    #    type        | in             | out         | supported
    #   --------------------------------------------------------
    #    None        | 0              | 1           | yes
    #    slice       | 1              | 1           | yes
    #    int         | 1              | 0           | yes
    #    ellipsis    | (dim - others) | same        | yes
    #    list[int]   | 1              | 1           | yes (matlab-style)
    #    list[bool]  | 1              | 1           | no
    #    array[int]  | 1              | array.dim() | no
    #    array[bool] | array.dim()    | 1           | no

    nb_dim_in = []
    nb_dim_out = []
    ind_ellipsis = None
    for n_ind, ind in enumerate(index):
        if ind is None:
            nb_dim_in.append(0)
            nb_dim_out.append(1)
        elif isinstance(ind, slice):
            nb_dim_in.append(1)
            nb_dim_out.append(1)
        elif ind is Ellipsis:
            if ind_ellipsis is not None:
                raise ValueError('Cannot have more than one ellipsis.')
            ind_ellipsis = n_ind
            nb_dim_in.append(-1)
            nb_dim_out.append(-1)
        elif is_int(ind):
            ind = torch.as_tensor(ind, dtype=torch.int64)
            if ind.dim() > 1:
                raise ValueError('Integer indices should be scalars or '
                                 'sequences. Got array with shape {}.'
                                 .format(ind.dim()))
            nb_dim_in.append(1)
            nb_dim_out.append(ind.dim())
            index[n_ind] = ind.item() if ind.dim() == 0 else ind.tolist()
        else:
            raise TypeError('Indices should be integers, slices '
                            'or ellipses. Got {}.'.format(type(ind)))

    # deal with ellipsis
    nb_known_dims = sum(n for n in nb_dim_in if n > 0)
    if ind_ellipsis is not None:
        nb_dim_in[ind_ellipsis] = max(0, nb_dim - nb_known_dims)
        nb_dim_out[ind_ellipsis] = nb_dim_in[ind_ellipsis]
    else:
        index.append(Ellipsis)
        nb_dim_in.append(max(0, nb_dim - nb_known_dims))
        nb_dim_out.append(nb_dim_in[-1])

    # transform each index into a slice
    nb_ind = 0
    index0 = index
    index = []
    output_shape = []
    for d, ind in enumerate(index0):
        if ind is None:
            # new axis (0 -> 1)
            output_shape.append(1)
            nb_ind += nb_dim_in[d]
            index.append(None)
        elif isinstance(ind, slice):
            # slice (1 -> 1)
            #   we pre-compute start/stop/step instead of leaving None
            start = ind.start
            stop = ind.stop
            step = ind.step
            step = 1 if step is None else step
            start = 0 if (start is None and step > 0) else \
                shape[nb_ind] - 1 if (start is None and step < 0) else \
                    shape[nb_ind] + start if start < 0 else \
                        start
            stop = shape[nb_ind] if (stop is None and step > 0) else \
                -1 if (stop is None and step < 0) else \
                    shape[nb_ind] + stop if stop < 0 else \
                        stop
            index.append(slice(start, stop, step))
            output_shape.append(min(shape[nb_ind], (stop - start) // step))
            nb_ind += nb_dim_in[d]
        elif ind is Ellipsis:
            # ellipsis (... -> ...)
            #   we replace one ellipsis with a series of slices
            for dd in range(nb_ind, nb_ind + nb_dim_in[d]):
                start = 0
                step = 1
                stop = shape[nb_ind]
                index.append(slice(start, stop, step))
                output_shape.append(shape[nb_ind])
                nb_ind += 1
        else:
            # scalar (1 -> 0) or list (1 -> 1)
            assert isinstance(ind, (int, list))  # already checked
            index.append(ind)
            if isinstance(ind, list):
                output_shape.append(len(ind))
            nb_ind += nb_dim_in[d]

    return tuple(index), tuple(output_shape)


def compose_index(parent, child):
    """Compose two sub-indexing

    Assume `_expand_index` has been called on parent and child before.
    """

    def oob(i):
        """Out-of-bound error."""
        raise IndexError('Index out-of-bound in parent dimension '
                         '{}.'.format(i))


    parent = list(parent)
    child = list(child)

    i_parent = -1
    new_parent = []
    while parent:
        # copy leading `None`s
        while child and child[0] is None:
            new_parent = [None, *new_parent]
            child = child[1:]

        # if no more children, just keep the remaining parents
        # (I don't think this should ever happen)
        if not child:
            new_parent += parent
            break

        # extract leading dimension
        p, *parent = parent
        c, *child = child
        i_parent += 1

        if isinstance(p, int) and p >= 0:
            # dropped axis
            continue

        elif p is None:
            # virtual axis
            if isinstance(c, int):
                if c > 0:
                    # out-of-bound
                    oob(i_parent)
                continue
            elif isinstance(c, list):
                # we keep the new axis
                # + with broadcast if more than one value
                #  (negative index == broadcast)
                if not all(idx == 0 for idx in c):
                    oob(i_parent)
                size = len(c)
                new_parent.append(None if size == 1 else -size)
                continue
            elif isinstance(c, slice):
                # keep the new axis
                if c.start != 0:
                    oob(i_parent)
                size = (c.stop - c.start)//c.step
                new_parent.append(None if size == 1 else -size)
                continue
            else:
                assert False, "p is None and c is {}".format(c)

        elif isinstance(p, int) and p < 0:
            # broadcasting
            if isinstance(c, int):
                if c >= -p:
                    # out-of-bound
                    oob(i_parent)
                continue
            elif isinstance(c, list):
                # change size of broadcast
                if not all(idx < -p for idx in c):
                    oob(i_parent)
                size = len(c)
                new_parent.append(None if size == 1 else -size)
                continue
            elif isinstance(c, slice):
                # keep the new axis
                if c.start >= -p:
                    oob(i_parent)
                size = (c.stop - c.start)//c.step
                new_parent.append(None if size == 1 else -size)
                continue
            else:
                assert False, "p is neg and c is {}".format(c)

        elif isinstance(p, list):
            size = len(p)
            if isinstance(c, int):
                if c >= p:
                    oob(i_parent)
                new_parent.append(p[c])
                continue
            elif isinstance(c, list):
                if not all(idx < p for idx in c):
                    oob(i_parent)
                new_parent.append([p[idx] for idx in c])
            elif isinstance(c, slice):
                if c.start >= size:
                    oob(i_parent)
                c = list((range(c.start, c.stop, c.step)))
                new_parent.append([p[idx] for idx in c if idx < size])

        elif isinstance(p, slice):
            # slice
            size = (p.stop - p.start)//p.step
            if isinstance(c, int):
                # convert to scalar
                if c >= size:
                    oob(i_parent)
                new_parent.append(p.start + c * p.step)
                continue
            elif isinstance(c, list):
                # convert to list
                if not all(idx < size for idx in c):
                    oob(i_parent)
                p = list((range(p.start, p.stop, p.step)))
                p = [p[idx] for idx in c]
                new_parent.append(p)
                continue
            elif isinstance(c, slice):
                # merge slices
                start = p.start + c.start * p.step
                stop = p.start + c.stop * p.step
                step = p.step * c.step
                new_parent.append(slice(start, stop, step))
                continue
            else:
                assert False, "p is slice and c is {}".format(c)

    return tuple(new_parent)


def invert_permutation(perm):
    """Return the inverse of a permutation

    Parameters
    ----------
    perm : sequence[int]
        Permutations. A permutation is a shuffled set of indices.

    Returns
    -------
    iperm : list[int]
        Inverse permutation.

    """
    perm = list(perm)
    iperm = list(range(len(perm)))
    iperm = [iperm[p] for p in range(len(perm))]
    return iperm
