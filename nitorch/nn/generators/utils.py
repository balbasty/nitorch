import inspect
import torch


def batch_callable(f):
    if not callable(f):
        return False
    return 'batch' in inspect.signature(f).parameters.keys()


def padshape(x, target_dim, dim=-1):
    """
    Insert singleton dimensions in position `dim` until the tensor as
    dimensionality `target_dim`.
    """
    pad_dim = max(0, target_dim - x.dim())
    old_shape = list(x.shape)
    new_shape = old_shape[:dim] + [1] * pad_dim + old_shape[dim:]
    x = x.reshape(new_shape)
    return x


def call_spatial(f, batch, dim):
    """
    Call/reshape appropriately the parameter or sampling function function.
    """
    if batch_callable(f):
        f = torch.as_tensor(f(batch))
        f = padshape(f, dim+2)
    elif callable(f):
        f = torch.as_tensor(f(batch))
        f = padshape(f, dim+1)
    else:
        f = padshape(torch.as_tensor(f), dim+1)
    return f
