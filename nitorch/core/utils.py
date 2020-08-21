import torch
from .pyutils import pad_list


def unsqueeze(input, dim=0, ndim=1):
    """Adds singleton dimensions to a tensor.

    This function expends `torch.unsqueeze` with additional options.

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
