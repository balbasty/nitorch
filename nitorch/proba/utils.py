from nitorch.core import utils as torch_utils
from . import variables


def to(*args, dtype=None, device=None):
    """Move/convert to a common dtype or device.

    Parameters
    ----------
    *args : tensor_like
        Input tensors or tensor-like objects
    dtype : str or torch.dtype, optional
        Target data type
    device : str or torch.device, optional
        Target device

    Returns
    -------
    *args : tensor_like
        Converted tensors

    """
    backend = dict(dtype=dtype, device=device)
    args = [arg.to(**backend) if isinstance(arg, variables.RandomVariable)
            else torch_utils.to(arg, **backend) for arg in args]
    if len(args) == 1:
        return args[0]
    else:
        return tuple(args)


def max_backend(*args):
    """Get the (max) dtype and device.

    Parameters
    ----------
    args : tensors

    Returns
    -------
    dict with keys 'dtype' and 'device'

    """
    return dict(dtype=max_dtype(*args), device=max_device(*args))


def max_device(*args):
    """Find a common device for all inputs.

    If at least one input object is on a CUDA device:
        * if all cuda object are on the same cuda device, return it
        * if some objects are on different cuda devices, return
          `device('cuda')` without an index.
    Else, return device('cpu') or None.

    Parameters
    ----------
    *args : tensor_like or device_like

    Returns
    -------
    device : torch.device

    """
    args = [arg.device if isinstance(arg, variables.RandomVariable) else arg
            for arg in args]
    return torch_utils.max_device(*args)


def max_dtype(*args, force_float=False):
    """Find the maximum data type from a series of inputs.

    The returned dtype is the best one to use for upcasting the objects.

        * Tensors and arrays have priority python objects.
        * Tensors and arrays with non-null dimensionality have priority
          over scalars.
        * If any of the torch/numpy objects have a floating point type
          a floating point type is returned.
        * If any of the objects is complex, a complex type is returned.
        * If all torch/numpy objects have an integer type and there is
          an integer type that avoids overflowing, it is returned.
        * If no integer type that ensures underflowing exists, the default
          floating point data type is returned.
        * If `force_float is True`, a floating point data type is returned
          even if all input objects have an integer data type.

    Parameters
    ----------
    *args : tensor_like or type_like
    force_float : bool, default=False

    Returns
    -------
    dtype : torch.dtype

    """
    args = [arg.dtype if isinstance(arg, variables.RandomVariable) else arg
            for arg in args]
    return torch_utils.max_dtype(*args, force_float=force_float)
