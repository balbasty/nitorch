"""PyTorch utilities."""

import torch
from .pyutils import make_list, make_tuple


def as_tensor(input, dtype=None, device=None):
    """Convert object to tensor.

    This function expands ``torch.as_tensor`` by accepting nested lists
    of tensors. It works by recursively stacking elements of the input
    list. It is probably much slower than ``torch.as_tensor``.

    Parameters
    ----------
    input : tensor_like
        Input object: tensor or (nested) list/tuple of tensors/scalars
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output tensor.

    """
    def _stack(x, dtype, device):
        if torch.is_tensor(x):
            return x.to(device if device is not None else x.device,
                        dtype if dtype is not None else x.dtype)
        else:
            if isinstance(x, (list, tuple)):
                dtype0, device0 = info(x)
                dtype0 = dtype if dtype is not None else dtype0
                device0 = device if device is not None else device0
                return torch.stack([_stack(e, dtype0, device0) for e in x])
            else:
                return torch.as_tensor(x, dtype=dtype, device=device)

    return _stack(input, dtype, device)


def unsqueeze(input, dim=0, ndim=1):
    """Adds singleton dimensions to a tensor.

    This function expands `torch.unsqueeze` with additional options.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int or list[int], default=0
        Position(s) at which to insert singleton dimensions.
    ndim : int or list[int], default=1
        Number of singleton dimensions inserted in each position.

    Returns
    -------
    output : tensor
        Tensor with additional singleton dimensions.
    """

    if not isinstance(dim, (list, tuple)):
        dim = [dim]
    if not isinstance(ndim, (list, tuple)):
        ndim = [ndim]
    ndim = make_list(ndim, len(dim))
    extra_dims = 0
    for d, nd in zip(dim, ndim):
        d += extra_dims
        for _ in range(nd):
            input = torch.unsqueeze(input, d)
        extra_dims += nd
    return input


def invert_permutation(perm):
    """Return the inverse of a permutation

    Parameters
    ----------
    perm : (..., N) tensor_like
        Permutations. A permutation is a shuffled set of indices.

    Returns
    -------
    iperm : (..., N) tensor
        Inverse permutation.

    Examples
    --------
    >>> import torch
    >>> from nitorch.core.utils import invert_permutation
    >>> perm = [0, 2, 3, 1]
    >>> a = torch.rand((len(perm),))
    >>> permuted_a = a[perm]
    >>> recovered_a = permuted_a[invert_permutation(perm)]
    >>> assert((a == recovered_a).all())

    """
    perm = torch.as_tensor(perm)
    shape = perm.shape
    device = perm.device
    perm = perm.reshape([-1, shape[-1]])
    n = perm.shape[-1]
    k = perm.shape[0]
    identity = torch.arange(n, dtype=torch.long, device=device)[None, ...]
    identity = identity.expand(k, n)  # Repeat without allocation
    iperm = torch.empty_like(perm).scatter_(-1, perm, identity)
    iperm = iperm.reshape(shape)
    return iperm


def info(*args):
    """Get the dtype and device of the first tensor of a list of objects."""
    for a in args:
        if torch.is_tensor(a):
            return a.dtype, a.device
    a = torch.as_tensor(args[0])
    return a.dtype, a.device


def same_storage(x, y):
    # type: (torch.Tensor, torch.Tensor) -> bool
    """Return true if `x` and `y` share the same underlying storage."""
    return x.storage().data_ptr() == y.storage().data_ptr()


def broadcast_backward(input, shape):
    """Sum a tensor across dimensions that have been broadcasted.

    Parameters
    ----------
    input : tensor
        Tensor with broadcasted shape.
    shape : tuple[int]
        Original shape.

    Returns
    -------
    output : tensor with shape `shape`

    """
    input_shape = input.shape
    dim = len(input_shape)
    for i, s in enumerate(reversed(shape)):
        dim = len(input_shape) - i - 1
        if s != input_shape[dim]:
            if s == 1:
                input = torch.sum(input, dim=dim, keepdim=True)
            else:
                raise ValueError('Shapes not compatible for broadcast: '
                                 '{} and {}'.format(tuple(input_shape), tuple(shape)))
    if dim > 0:
        input = torch.sum(input, dim=list(range(dim)), keepdim=False)
    return input


def requires_grad(ctx, name):
    """Checks if a named variable requires gradients."""
    for g, n in zip(ctx.needs_input_grad, ctx.names):
        if n == name:
            return g
    return False


def slice_tensor(x, index, dim=None):
    """Index a tensor along one or several dimensions.

    This function is relatively similar to `torch.index_select`, except
    that it uses the native indexing mechanism and can therefore
    returns a tensor that use the same storage as the input tensor.

    Parameters
    ----------
    x : tensor
        Input tensor.
    index : index_like or tuple[index_like]
        Indices to select along each dimension in `dim`.
        If multiple dimensions are indexed, they *must* be held in a
        tuple (not a list). Each index can be a long, list of long,
        slice or tensor of long, but *cannot* be an ellipsis or
        tensor of bool.
    dim : int or sequence[int], optional
        Dimensions to index. If it is a list, `index` *must* be a tuple.
        By default, the last `n` dimensions (where `n` is the number of
        indices in `index`) are used.


    Returns
    -------
    y : tensor
        Output tensor.

    """
    # format (dim, index) as (list, tuple) with same length
    if not isinstance(index, tuple):
        index = (index,)
    if dim is None:
        dim = list(range(-len(index), 0))
    dim = make_list(dim)
    nb_dim = max(len(index), len(dim))
    dim = make_list(dim, nb_dim)
    index = make_tuple(index, nb_dim)

    # build index
    full_index = [slice(None)] * x.dim()
    for d, ind in zip(dim, index):
        if ind is Ellipsis or (torch.is_tensor(ind) and
                               ind.dtype == torch.bool):
            raise TypeError('`index` cannot be an ellipsis or mask')
        full_index[d] = ind
    full_index = tuple(full_index)

    return x.__getitem__(full_index)


def broadcast_to(*tensors):
    """Broadcast to a given shape.

    Parameters
    ----------
    *tensors : any number of tensors
    shape : list[int]
        Target shape that must be compatible with all tensors
        according to :ref:`broadcasting-semantics`.

    Returns
    -------
    *tensors : tensors 'reshaped' to shape.

    .. warning::
        This function makes use of zero strides, so more than
        one output values can point to the same memory location.
        It is advided not too write in these tensors.

    """
    *tensors, shape = tensors
    tensors = [torch.as_tensor(tensor) for tensor in tensors]
    for i, tensor in enumerate(tensors):
        # 1. pad with singleton dimensions on the left
        if len(shape) > tensor.dim():
            tensor = unsqueeze(tensor, dim=0, ndim=len(shape)-tensor.dim())
        elif len(shape) < tensor.dim():
            raise ValueError('Cannot broadcast shape {} to {}: '
                             'the target shape has less dimensions that '
                             'the input shape.'.format(tensor.shape, shape))
        # 2. zero-stride singleton dimensions
        strides = list(tensor.stride())
        for d in range(len(shape)):
            if tensor.shape[d] != shape[d]:
                if tensor.shape[d] == 1:
                    strides[d] = 0
                else:
                    raise ValueError('Cannot broadcast shape {} to {}: '
                                     'shapes have different non-singleton '
                                     'dimensions.'.format(tensor.shape, shape))
        tensor = tensor.as_strided(shape, strides)
        tensors[i] = tensor

    # return
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tuple(tensors)
