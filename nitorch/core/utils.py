import torch
from .pyutils import pad_list


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
                return torch.stack([_stack(e, dtype, device) for e in x])
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
    ndim = pad_list(ndim, len(dim))
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
