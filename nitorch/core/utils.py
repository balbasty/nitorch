"""PyTorch utilities."""

import torch
from . import py
from .py import make_list, make_tuple, ensure_list
from .constants import inf, eps
from .dtypes import as_torch as dtype_astorch
from . import dtypes
from nitorch._C.grid import GridCount, GridPull, GridPush
from .optionals import numpy as np
import numbers
import os
import random
from typing import Optional
Tensor = torch.Tensor


def _compare_versions(version1, mode, version2):
    for v1, v2 in zip(version1, version2):
        if mode in ('gt', '>'):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('ge', '>='):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('lt', '<'):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
        elif mode in ('le', '<='):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
    if mode in ('gt', 'lt', '>', '<'):
        return False
    else:
        return True


def torch_version(mode, version):
    """Check torch version

    Parameters
    ----------
    mode : {'<', '<=', '>', '>='}
    version : tuple[int]

    Returns
    -------
    True if "torch.version <mode> version"

    """
    current_version, *cuda_variant = torch.__version__.split('+')
    major, minor, patch, *_ = current_version.split('.')
    # strip alpha tags
    for x in 'abcdefghijklmnopqrstuvwxy':
        if x in patch:
            patch = patch[:patch.index(x)]
    current_version = (int(major), int(minor), int(patch))
    version = py.make_list(version)
    return _compare_versions(current_version, mode, version)


def reproducible(seed=1234):
    """Ensure reproducible results.

    Parameters
    ----------
    seed : int, default=1234
        Seed for random number generators.

    """	
    random.seed(seed)
    if np:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


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
    # TODO: if torch >= 1.6, use` torch.as_tensor`
    #   I have to clean uses of `utils.as_tensor` first because the
    #   order of arguments is a bit different (I think it is device then
    #   dtype in torch)
    def _stack(x, dtype, device):
        if torch.is_tensor(x):
            return x.to(device if device is not None else x.device,
                        dtype if dtype is not None else x.dtype)
        else:
            if isinstance(x, (list, tuple)):
                subs = [_stack(e, dtype, device) for e in x]
                backend = max_backend(*subs)
                subs = [elem.to(**backend) for elem in subs]
                return torch.stack(subs)
            else:
                return torch.as_tensor(x, dtype=dtype, device=device)

    return _stack(input, dtype, device)


def make_vector(input, n=None, crop=True, *args, 
                dtype=None, device=None, **kwargs):
    """Ensure that the input is a (tensor) vector and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output vector.

    """
    input = as_tensor(input, dtype=dtype, device=device).flatten()
    if n is None:
        return input
    if n is not None and input.numel() >= n:
        return input[:n] if crop else input
    has_default = False
    if args:
        has_default = True
        default = args[0]
    elif 'default' in kwargs:
        has_default = True
        default = kwargs['default']
    if has_default:
        return ensure_shape(input, n, mode='constant', value=default)
    else:
        return ensure_shape(input, n, mode='replicate')
        

def unsqueeze(input, dim=0, ndim=1):
    """Adds singleton dimensions to a tensor.

    This function expands `torch.unsqueeze` with additional options.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int, default=0
        Position at which to insert singleton dimensions.
    ndim : int, default=1
        Number of singleton dimensions to insert.

    Returns
    -------
    output : tensor
        Tensor with additional singleton dimensions.
    """
    for _ in range(ndim):
        input = torch.unsqueeze(input, dim)
    return input


def squeeze(input, dim=0, ndim=1):
    """Removes singleton dimensions to a tensor.

    This function expands `torch.squeeze` with additional options.

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int, default=0
        Position at which to drop singleton dimensions.
    ndim : int, default=1
        Number of singleton dimensions to drop.

    Returns
    -------
    output : tensor
        Tensor with singleton dimensions removed.
    """
    for _ in range(ndim):
        input = torch.squeeze(input, dim)
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


def shiftdim(x, n=None):
    """Shift the dimensions of x by n.

    Parameters
    ----------
        x : torch.Tensor
            Input tensor.
        n : int, default=None
            Shift.
            * When N is positive, `shiftdim` shifts the dimensions to
              the left and wraps the N leading dimensions to the end.
            * When N is negative, `shiftdim` shifts the dimensions to
              the right and pads with singletons.
            * When N is None, `shiftdim` removes all leading singleton
              dimensions. The number of removed dimensions is returned
              as well.

    Returns
    -------
        x : torch.Tensor
            Output tensor.
        n : int, if n is None
            Number of removed dimensions

    """
    if n is None:
        shape = torch.as_tensor(x.size())
        n = (shape != 1).nonzero()
        if n.numel() == 0:
            n = x.dim()
            x = x.reshape([])
        else:
            n = n[0]
            x = x.reshape(shape[n:].tolist())
        return x, n
    elif n < 0:
        x = x.reshape((1,)*(-n) + x.size())
    elif n > 0:
        n = n % x.dim()
        x = x.permute(tuple(range(n, x.dim())) + tuple(range(n)))
    return x


def fast_movedim(input, source, destination):
    """Move the position of exactly one dimension"""
    dim = input.dim()

    source = dim + source if source < 0 else source
    destination = dim + destination if destination < 0 else destination
    permutation = list(range(dim))
    del permutation[source]
    permutation.insert(destination, source)
    return input.permute(*permutation)


def movedim(input, source, destination):
    """Moves the position of one or more dimensions

    Other dimensions that are not explicitly moved remain in their
    original order and appear at the positions not specified in
    destination.

    Parameters
    ----------
    input : tensor
        Input tensor
    source : int or sequence[int]
        Initial positions of the dimensions
    destination : int or sequence[int]
        Output positions of the dimensions.

        If a single destination is provided:
        - if it is negative, the last source dimension is moved to
          `destination` and all other source dimensions are moved to its left.
        - if it is positive, the first source dimension is moved to
          `destination` and all other source dimensions are moved to its right.

    Returns
    -------
    output : tensor
        Tensor with moved dimensions.

    """
    input = torch.as_tensor(input)
    dim = input.dim()
    source = make_list(source)
    destination = make_list(destination)
    if len(destination) == 1:
        # we assume that the user wishes to keep moved dimensions
        # in the order they were provided
        destination = destination[0]
        if destination >= 0:
            destination = list(range(destination, destination+len(source)))
        else:
            destination = list(range(destination+1-len(source), destination+1))
    if len(source) != len(destination):
        raise ValueError('Expected as many source as destination positions.')
    source = [dim + src if src < 0 else src for src in source]
    destination = [dim + dst if dst < 0 else dst for dst in destination]
    if len(set(source)) != len(source):
        raise ValueError(f'Expected source positions to be unique but got '
                         f'{source}')
    if len(set(destination)) != len(destination):
        raise ValueError(f'Expected destination positions to be unique but got '
                         f'{destination}')

    # compute permutation
    positions_in = list(range(dim))
    positions_out = [None] * dim
    for src, dst in zip(source, destination):
        positions_out[dst] = src
        positions_in[src] = None
    positions_in = filter(lambda x: x is not None, positions_in)
    for i, pos in enumerate(positions_out):
        if pos is None:
            positions_out[i], *positions_in = positions_in

    return input.permute(*positions_out)


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
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype, device=device)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype, device=device)
                     if arg is not None else arg for arg in args)


def to_max_backend(*args, force_float=False, dtype=None, device=None):
    """Move to a common dtype and device.

    See `max_dtype` and `max_device`.

    Parameters
    ----------
    *args : tensor_like
    force_float : bool, default=False

    Returns
    -------
    *args_to : tensor

    """
    if len(args) == 0:
        return
    dtype = max_dtype(*args, dtype, force_float=force_float)
    device = max_device(*args, device)
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype, device=device)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype, device=device)
                     if arg is not None else arg for arg in args)


def to_max_device(*args):
    """Move to a common device.

    See `max_device`.

    Parameters
    ----------
    *args : tensor_like

    Returns
    -------
    *args_to : tensor

    """
    if len(args) == 0:
        return
    device = max_device(*args)
    if len(args) == 1:
        return torch.as_tensor(args[0], device=device)
    else:
        return tuple(torch.as_tensor(arg, device=device)
                     for arg in args)


def to_max_dtype(*args):
    """Move to a common data type.

    See `max_dtype`.

    Parameters
    ----------
    *args : tensor_like

    Returns
    -------
    *args_to : tensor

    """
    if len(args) == 0:
        return
    dtype = max_dtype(*args)
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype)
                     for arg in args)


def backend(x):
    """Return the backend (dtype and device) of a tensor

    Parameters
    ----------
    x : tensor

    Returns
    -------
    dict with keys 'dtype' and 'device'

    """
    return dict(dtype=x.dtype, device=x.device)


def max_backend(*args, dtype=None, device=None):
    """Get the (max) dtype and device.

    Parameters
    ----------
    args : tensors

    Returns
    -------
    dict with keys 'dtype' and 'device'

    """
    return dict(dtype=max_dtype(*args, dtype),
                device=max_device(*args, device))


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
    from .optionals import numpy as np
    is_array = lambda x: (isinstance(x, np.ndarray) if np else False)
    is_tensor = torch.is_tensor

    def select_device(*many_devices):
        if len(many_devices) == 0:
            return None
        elif len(many_devices) == 1:
            return many_devices[0]
        device1, device2, *many_devices = many_devices
        if len(many_devices) > 0:
            return select_device(select_device(device1, device2), *many_devices)
        if device1 is None:
            return device2
        elif device2 is None:
            return device1
        elif device1.type == 'cuda' and device2.type != 'cuda':
            return device1
        elif device2.type == 'cuda' and device1.type != 'cuda':
            return device2
        elif device1.index is None:
            return device2
        elif device2.index is None:
            return device1
        elif device1.index == device2.index:
            return device1
        else:
            return torch.device('cuda')

    def explore_device(x):
        if x is None:
            return None
        if isinstance(x, (torch.device, str)):
            return torch.device(x)
        elif is_tensor(x):
            return x.device
        elif is_array(x) or isinstance(x, numbers.Number):
            # numpy/builtin type: None
            return None
        else:
            # assume it is a sequence: check what we find in there
            devices = [explore_device(elem) for elem in x]
            return select_device(*devices)

    return explore_device(args)


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
    from .optionals import numpy as np
    is_array = lambda x: (isinstance(x, np.ndarray) if np else False)
    is_tensor = torch.is_tensor
    is_np_dtype = lambda x: ((isinstance(x, np.dtype) or
                                 (isinstance(x, type) and
                                  issubclass(x, np.number)))
                                if np else False)
    is_torch_dtype = lambda x: isinstance(x, torch.dtype)
    is_py_dtype = lambda x: isinstance(x, type) and issubclass(x, numbers.Number)
    is_dtype = lambda x: is_torch_dtype(x) or is_np_dtype(x) or is_py_dtype(x)

    def upcast(*many_types):
        if len(many_types) == 0:
            return None
        elif len(many_types) == 1:
            return many_types[0]
        dtype1, dtype2, *many_types = many_types
        if len(many_types) > 0:
            return upcast(upcast(dtype1, dtype2), *many_types)
        # here, we only have torch dtypes
        if dtype1 is None:
            return dtype2
        elif dtype2 is None:
            return dtype1
        elif dtype1 is torch.complex128 or dtype2 is torch.complex128:
            return torch.complex128
        elif dtype1 is torch.complex64 or dtype2 is torch.complex64:
            return torch.complex64
        elif dtype1 is torch.complex32 or dtype2 is torch.complex32:
            return torch.complex32
        elif dtype1 is torch.float64 or dtype2 is torch.float64:
            return torch.float64
        elif dtype1 is torch.float32 or dtype2 is torch.float32:
            return torch.float32
        elif dtype1 is torch.float16 or dtype2 is torch.float16:
            return torch.float16
        elif dtype1 is torch.int64 or dtype2 is torch.int64:
            return torch.int64
        elif dtype1 is torch.int32 or dtype2 is torch.int32:
            return torch.int32
        elif dtype1 is torch.int16 or dtype2 is torch.int16:
            return torch.int16
        elif dtype1 is torch.int8 and dtype2 is torch.int8:
            return torch.int8
        elif dtype1 is torch.uint8 and dtype2 is torch.uint8:
            return torch.uint8
        elif (dtype1 is torch.int8 and dtype2 is torch.uint8) or \
             (dtype1 is torch.uint8 and dtype2 is torch.int8):
            return torch.int16
        elif dtype1 is torch.bool and dtype2 is torch.bool:
            return torch.bool
        else:
            raise TypeError('We do not deal with type {} or {} yet.'
                            .format(dtype1, dtype2))

    def explore_dtype(x, n_pass=1):
        # find the max data type at a given pass
        if x is None:
            return None
        elif is_dtype(x):
            return dtype_astorch(x)
        elif (is_tensor(x) or is_array(x)) and len(x.shape) > 0:
            return dtype_astorch(x.dtype)
        elif is_tensor(x) or is_array(x):
            # scalar: only return if pass 2+
            return dtype_astorch(x.dtype) if n_pass >= 2 else None
        elif isinstance(x, numbers.Number):
            # builtin type:  only return if pass 3+
            return dtype_astorch(type(x)) if n_pass >= 3 else None
        else:
            # assume it is a sequence: check what we find in there
            dtypes = [explore_dtype(elem, n_pass) for elem in x]
            return upcast(*dtypes)

    # 1) tensors/arrays with dim > 0
    maxdtype = explore_dtype(args, n_pass=1)

    # 2) tensor/arrays with dim == 0
    if maxdtype is None:
        maxdtype = upcast(maxdtype, explore_dtype(args, n_pass=2))

    # 3) tensor/arrays
    if maxdtype is None:
        maxdtype = upcast(maxdtype, explore_dtype(args, n_pass=3))

    # Finally) ensure float
    if force_float:
        maxdtype = upcast(maxdtype, torch.get_default_dtype())

    return maxdtype


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


def fast_slice_tensor(x, index, dim=-1):
    """Index a tensor along one dimensions.

    This function is relatively similar to `torch.index_select`, except
    that it uses the native indexing mechanism and can therefore
    returns a tensor that use the same storage as the input tensor.

    It is faster but less versatile than `slice_tensor`.

    Parameters
    ----------
    x : tensor
        Input tensor.
    index : int or list[int] or slice
        Indices to select along `dim`.
    dim : int, default=last
        Dimension to index.

    Returns
    -------
    y : tensor
        Output tensor.

    """
    slicer = [slice(None)] * x.dim()
    slicer[dim] = index
    slicer = tuple(slicer)
    return x[slicer]


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
    dim = ensure_list(dim)
    nb_dim = max(len(index), len(dim))
    dim = ensure_list(dim, nb_dim)
    index = tuple(ensure_list(index, nb_dim))

    # build index
    full_index = [slice(None)] * x.dim()
    for d, ind in zip(dim, index):
        if ind is Ellipsis or (torch.is_tensor(ind) and
                               ind.dtype == torch.bool):
            raise TypeError('`index` cannot be an ellipsis or mask')
        full_index[d] = ind
    full_index = tuple(full_index)

    return x.__getitem__(full_index)


def max_shape(*shapes, side='left'):
    """Compute maximum (= broadcasted) shape.

    Parameters
    ----------
    *shapes : sequence[int]
        any number of shapes
    side : {'left', 'right'}, default='left'
        Side to add singleton dimensions.

    Returns
    -------
    shape : tuple[int]
        Maximum shape

    """
    def error(s0, s1):
        raise ValueError('Incompatible shapes for broadcasting: {} and {}.'
                         .format(s0, s1))

    # 1. nb dimensions
    nb_dim = 0
    for shape in shapes:
        nb_dim = max(nb_dim, len(shape))

    # 2. pad with singleton dimensions
    max_shape = [1] * nb_dim
    for i, shape in enumerate(shapes):
        pad_size = nb_dim - len(shape)
        ones = [1] * pad_size
        if side == 'left':
            shape = [*ones, *shape]
        else:
            shape = [*shape, *ones]
        max_shape = [max(s0, s1) if s0 == 1 or s1 == 1 or s0 == s1
                     else error(s0, s1) for s0, s1 in zip(max_shape, shape)]

    return max_shape


def expanded_shape(*shapes, side='left'):
    """Expand input shapes according to broadcasting rules

    Parameters
    ----------
    *shapes : sequence[int]
        Input shapes
    side : {'left', 'right'}, default='left'
        Side to add singleton dimensions.

    Returns
    -------
    shape : tuple[int]
        Output shape

    Raises
    ------
    ValueError
        If shapes are not compatible for broadcast.

    """
    def error(s0, s1):
        raise ValueError('Incompatible shapes for broadcasting: {} and {}.'
                         .format(s0, s1))

    # 1. nb dimensions
    nb_dim = 0
    for shape in shapes:
        nb_dim = max(nb_dim, len(shape))

    # 2. enumerate
    shape = [1] * nb_dim
    for i, shape1 in enumerate(shapes):
        pad_size = nb_dim - len(shape1)
        ones = [1] * pad_size
        if side == 'left':
            shape1 = [*ones, *shape1]
        else:
            shape1 = [*shape1, *ones]
        shape = [max(s0, s1) if s0 == 1 or s1 == 1 or s0 == s1
                 else error(s0, s1) for s0, s1 in zip(shape, shape1)]

    return tuple(shape)


def expand(*tensors, side='left', dry_run=False, **kwargs):
    """Broadcast to a given shape.

    Parameters
    ----------
    *tensors : tensor
        any number of tensors
    shape : list[int]
        Target shape that must be compatible with all tensors
        according to :ref:`broadcasting-semantics`.
    side : {'left', 'right'}, default='left'
        Side to add singleton dimensions.
    dry_run : bool, default=False
        Return the broadcasted shape instead of the broadcasted tensors.

    Returns
    -------
    *tensors : tensors 'reshaped' to shape.

    Raises
    ------
    ValueError
        If shapes are not compatible for broadcast.

    .. warning::
        This function makes use of zero strides, so more than
        one output values can point to the same memory location.
        It is advised not to write in these tensors.

    """
    if 'shape' in kwargs:
        shape = kwargs['shape']
    else:
        *tensors, shape = tensors
    tensors = [torch.as_tensor(tensor) for tensor in tensors]

    # -------------
    # Compute shape
    # -------------
    shape = expanded_shape(shape, *(t.shape for t in tensors), side=side)
    nb_dim = len(shape)

    if dry_run:
        return tuple(shape)

    # -----------------
    # Broadcast tensors
    # -----------------

    pad_dim = 0 if side == 'left' else -1
    for i, tensor in enumerate(tensors):
        # 1. pad with singleton dimensions on the left
        tensor = unsqueeze(tensor, dim=pad_dim, ndim=nb_dim-tensor.dim())
        # 2. expand tensor
        tensors[i] = tensor.expand(shape)

    if len(tensors) == 1:
        return tensors[0]
    else:
        return tuple(tensors)


# %% Padding
# This section defines boundary conditions that allow datapoints outside
# the field of view to be extrapolated.


def _bound_circular(i, n):
    return i % n


def _bound_replicate(i, n):
    return i.clamp(min=0, max=n-1)


def _bound_reflect2(i, n):
    n2 = n*2
    pre = (i < 0)
    i[pre] = n2 - 1 - ((-i[pre]-1) % n2)
    i[~pre] = (i[~pre] % n2)
    post = (i >= n)
    i[post] = n2 - i[post] - 1
    return i


def _bound_reflect1(i, n):
    if n == 1:
        return torch.zeros(i.size(), dtype=i.dtype, device=i.device)
    else:
        n2 = (n-1)*2
        pre = (i < 0)
        i[pre] = -i[pre]
        i = i % n2
        post = (i >= n)
        i[post] = n2 - i[post]
        return i


def ensure_shape(inp, shape, mode='constant', value=0, side='post'):
    """Pad/crop a tensor so that it has a given shape

    Parameters
    ----------
    inp : tensor
        Input tensor
    shape : sequence
        Output shape
    mode : 'constant', 'replicate', 'reflect1', 'reflect2', 'circular'
        default='constant'
    value : scalar, default=0
        Value for mode 'constant'
    side : {'pre', 'post', 'both'}, default='post'
        Side to crop/pad

    Returns
    -------
    out : tensor
        Padded tensor with shape `shape`

    """
    inp = torch.as_tensor(inp)
    shape = make_list(shape)
    shape = shape + [1] * max(0, inp.dim() - len(shape))
    if inp.dim() < len(shape):
        inp = inp.reshape(inp.shape + (1,) * max(0, len(shape) - inp.dim()))
    inshape = inp.shape
    shape = [inshape[d] if shape[d] is None else shape[d]
             for d in range(len(shape))]
    ndim = len(shape)

    # crop
    if side == 'both':
        crop = [max(0, inshape[d] - shape[d]) for d in range(ndim)]
        index = tuple(slice(c//2, (c//2 - c) or None) for c in crop)
    elif side == 'pre':
        crop = [max(0, inshape[d] - shape[d]) for d in range(ndim)]
        index = tuple(slice(-c or None) for c in crop)
    else:  # side == 'post'
        index = tuple(slice(min(shape[d], inshape[d])) for d in range(ndim))
    inp = inp[index]

    # pad
    pad_size = [max(0, shape[d] - inshape[d]) for d in range(ndim)]
    if side == 'both':
        pad_size = [[p//2, p-p//2] for p in pad_size]
        pad_size = [q for p in pad_size for q in p]
        side = None
    inp = pad(inp, tuple(pad_size), mode=mode, value=value, side=side)

    return inp


_bounds = {
    'circular': _bound_circular,
    'replicate': _bound_replicate,
    'reflect': _bound_reflect1,
    'reflect1': _bound_reflect1,
    'reflect2': _bound_reflect2,
    }
_bounds['dft'] = _bounds['circular']
_bounds['dct2'] = _bounds['reflect2']
_bounds['dct1'] = _bounds['reflect1']


_modifiers = {
    'circular': lambda x, i, n: x,
    'replicate': lambda x, i, n: x,
    'reflect': lambda x, i, n: x,
    'reflect1': lambda x, i, n: x,
    'reflect2': lambda x, i, n: x,
    }
_modifiers['dft'] = _modifiers['circular']
_modifiers['dct2'] = _modifiers['reflect2']
_modifiers['dct1'] = _modifiers['reflect1']


def pad(inp, padsize, mode='constant', value=0, side=None):
    """Pad a tensor.

    This function is a bit more generic than torch's native pad, but probably
    a bit slower:
        - works with any input type
        - works with arbitrarily large padding size
        - crops the tensor for negative padding values
        - implements additional padding modes
    When used with defaults parameters (side=None), it behaves
    exactly like `torch.nn.functional.pad`

    Boundary modes are:
        - 'circular' or 'dft'
        - 'reflect' or 'reflect1' or 'dct1'
        - 'reflect2' or 'dct2'
        - 'replicate'
        - 'constant'

    Side modes are 'pre', 'post', 'both' or None. If side is not None,
    inp.dim() values (or less) should be provided. If side is None,
    twice as many values should be provided, indicating different padding sizes
    for the 'pre' and 'post' sides. If the number of padding values is less
    than the dimension of the input tensor, zeros are prepended.

    Parameters
    ----------
    inp : tensor_like
        Input tensor
    padsize : [sequence of] int
        Amount of padding in each dimension.
    mode : {'constant', 'replicate', 'reflect1', 'reflect2', 'circular'}, default='constant'
        Padding mode
    value : scalar, default=0
        Value to pad with in mode 'constant'.
    side : {'left', 'right', 'both', None}, default=None
        Use padsize to pad on left side ('pre'), right side ('post') or
        both sides ('both'). If None, the padding side for the left and
        right sides should be provided in alternate order.

    Returns
    -------
    tensor
        Padded tensor.

    """
    # Argument checking
    if mode not in tuple(_bounds.keys()) + ('constant', 'zero', 'zeros'):
        raise ValueError('Padding mode should be one of {}. Got {}.'
                         .format(tuple(_bounds.keys()) + ('constant',), mode))
    padsize = tuple(padsize)
    if not side:
        if len(padsize) % 2:
            raise ValueError('Padding length must be divisible by 2')
        padpre = padsize[::2]
        padpost = padsize[1::2]
    else:
        side = side.lower()
        if side == 'both':
            padpre = padsize
            padpost = padsize
        elif side in ('pre', 'left'):
            padpre = padsize
            padpost = (0,) * len(padpre)
        elif side in ('post', 'right'):
            padpost = padsize
            padpre = (0,) * len(padpost)
        else:
            raise ValueError(f'Unknown side `{side}`')
    padpre = (0,) * max(0, inp.dim()-len(padpre)) + padpre
    padpost = (0,) * max(0, inp.dim()-len(padpost)) + padpost
    if inp.dim() != len(padpre) or inp.dim() != len(padpost):
        raise ValueError('Padding length too large')

    padpre = torch.as_tensor(padpre)
    padpost = torch.as_tensor(padpost)

    # Pad
    if mode in ('zero', 'zeros'):
        mode, value = ('constant', 0)
    if mode == 'constant':
        return _pad_constant(inp, padpre, padpost, value)
    else:
        bound = _bounds[mode]
        modifier = _modifiers[mode]
        return _pad_bound(inp, padpre, padpost, bound, modifier)


def _pad_constant(inp, padpre, padpost, value):
    padpre = padpre.tolist()
    padpost = padpost.tolist()
    new_shape = [s + pre + post
                 for s, pre, post in zip(inp.shape, padpre, padpost)]
    out = inp.new_full(new_shape, value)
    slicer = [slice(pre, pre + s) for pre, s in zip(padpre, inp.shape)]
    out[tuple(slicer)] = inp
    return out


def _pad_bound(inp, padpre, padpost, bound, modifier):
    begin = -padpre
    end = tuple(d+p for d, p in zip(inp.size(), padpost))
    idx = tuple(range(b, e) for (b, e) in zip(begin, end))
    idx = tuple(bound(torch.as_tensor(i, device=inp.device),
                      torch.as_tensor(n, device=inp.device))
                for (i, n) in zip(idx, inp.shape))
    for d in range(inp.dim()):
        inp = inp.index_select(d, idx[d])
    return inp


def channel2last(tensor):
    """Warps: Channel to Last dimension order.

    . Channel ordering is: (Batch, Channel, X, Y, Z)
    . Last ordering is: (Batch, X, Y, Z, Channel))
    """
    tensor = torch.as_tensor(tensor)
    tensor = tensor.permute((0,) + tuple(range(2, tensor.dim())) + (1,))
    return tensor


def last2channel(tensor):
    """Warps: Last to Channel dimension order.

    . Channel ordering is: (Batch, Channel, X, Y, Z)
    . Last ordering is: (Batch, X, Y, Z, Channel))
    """
    tensor = torch.as_tensor(tensor)
    tensor = tensor.permute((0, - 1) + tuple(range(1, tensor.dim()-1)))
    return tensor


def isin(tensor, labels):
    """Returns a mask for elements that belong to labels

    Parameters
    ----------
    tensor : (*shape_tensor) tensor_like
        Input tensor
    labels : (*shape_labels, nb_labels) tensor_like
        Labels.
        `shape_labels` and `shape_tensor` should be broadcastable.

    Returns
    -------
    mask : (*shape) tensor[bool]

    """

    tensor = torch.as_tensor(tensor)
    labels = torch.as_tensor(labels)

    if labels.shape[-1] == 1:
        # only one label in the list
        return tensor == labels[..., 0]

    mask = tensor.new_zeros(tensor.shape, dtype=torch.bool)
    for label in torch.unbind(labels, dim=-1):
        mask = mask | (tensor == label)

    return mask


def ceil_pow(t, p=2.0, l=2.0, mx=None):
    """Ceils each element in vector t to the
    closest n that satisfies: l*p**n.

    This function is useful, for example, to ensure an image's dimensions
    work well in an encoding/decoding architecture.

    Parameters
    ----------
    t : (d, ), tensor
    p : float, default=2.0
    l : float, default=2.0
    mx : float, optional

    Returns
    ----------
    ct : (d, ), tensor

    """
    ct = t.clone()  # Do not modify in-place
    device = ct.device
    dtype0 = ct.dtype
    dtype = torch.float32
    dim = torch.as_tensor(ct, dtype=dtype, device=device)
    ct.clamp_max_(mx)
    d = len(ct)
    # Build array of l*p**[0, ..., N]
    N = 32
    p = torch.tensor(l, dtype=dtype, device=device) \
        * torch.tensor(p, dtype=dtype, device=device) \
        ** torch.arange(0, N, dtype=dtype, device=device)
    p = p.repeat(d, 1)
    # Ensure we ceil
    for n in range(d):
        p[n, p[n, ...] < ct[n]] = -inf
    ct = ct[..., None]
    # Find closest indices
    ix = torch.min((p - ct).abs(), dim=1)[1]
    ct = ct.squeeze()
    # Ceil input
    for n in range(d):
        if torch.isfinite(p[n, ix[n]]):
            ct[n] = p[n, ix[n]]
    # Return same datatype
    ct = ct.type(dtype0)

    return ct


def sub2ind(subs, shape, out=None):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D, ...) tensor_like
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) vector_like
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    ind : (...) tensor
        Linear indices
    """
    *subs, ind = subs
    if out is None:
        ind = torch.as_tensor(ind).clone()
    else:
        out.reshape(ind.shape).copy_(ind)
        ind = out
    bck = backend(ind)
    stride = py.cumprod(shape[1:], reverse=True)
    for i, s in zip(subs, stride):
        ind += torch.as_tensor(i, **bck) * torch.as_tensor(s, **bck)
    return ind


# floor_divide returns wrong results for negative values, because it truncates
# instead of performing a proper floor. In recent version of pytorch, it is
# advised to use div(..., rounding_mode='trunc'|'floor') instead.
# Here, we only use floor_divide on positive values so we do not care.
_trunc_div = ((lambda *a, **k: torch.div(*a, **k, rounding_mode='trunc'))
              if torch_version('>=', (1, 8)) else torch.floor_divide
              if torch_version('>=', (1, 5)) else (lambda x, y, **k: x // y))


def ind2sub(ind, shape, out=None):
    """Convert linear indices into sub indices (i, j, k).

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    ind : tensor_like
        Linear indices
    shape : (D,) vector_like
        Size of each dimension.
    out : tensor, optional
        Output placeholder

    Returns
    -------
    subs : (D, ...) tensor
        Sub-indices.
    """
    ind = torch.as_tensor(ind)
    bck = backend(ind)
    stride = py.cumprod(shape, reverse=True, exclusive=True)
    stride = torch.as_tensor(stride, **bck)
    if out is None:
        sub = ind.new_empty([len(shape), *ind.shape])
    else:
        sub = out.reshape([len(shape), *ind.shape])
    sub[:, ...] = ind
    for d in range(len(shape)):
        if d > 0:
            torch.remainder(sub[d], torch.as_tensor(stride[d-1], **bck), out=sub[d])
        sub[d] = _trunc_div(sub[d], stride[d], out=sub[d])
    return sub


def unfold(inp, kernel_size, stride=None, collapse=False):
    """Extract patches from a tensor.

    Parameters
    ----------
    inp : (..., *spatial) tensor
        Input tensor.
    kernel_size : [sequence of] int
        Patch shape.
    stride : [sequence of] int, default=`kernel_size`
        Stride.
    collapse : bool or 'view', default=False
        Collapse the original spatial dimensions.
        If 'view', forces collapsing to use the view mechanism, which ensures
        that no data copy is triggered. This can fail if the tensor's
        strides do not allow these dimensions to be collapsed.

    Returns
    -------
    out : (..., *spatial_out, *kernel_size) tensor
        Output tensor of patches.
        If `collapse`, the output spatial dimensions (`spatial_out`)
        are flattened.

    """
    inp = torch.as_tensor(inp)
    kernel_size = py.make_list(kernel_size)
    dim = len(kernel_size)
    batch_dim = inp.dim() - dim
    stride = py.make_list(stride, dim)
    stride = [st or sz for st, sz in zip(stride, kernel_size)]
    for d, (sz, st) in enumerate(zip(kernel_size, stride)):
        inp = inp.unfold(dimension=batch_dim+d, size=sz, step=st)
    if collapse:
        batch_shape = inp.shape[:-dim*2]
        if collapse == 'view':
            inp = inp.view([*batch_shape, -1, *kernel_size])
        else:
            inp = inp.reshape([*batch_shape, -1, *kernel_size])
    return inp


def fold(inp, dim=None, stride=None, shape=None, collapsed=False,
         reduction='mean'):
    """Reconstruct a tensor from patches.

    .. warning: This function only works if `kernel_size <= 2*stride`.

    Parameters
    ----------
    inp : (..., *spatial, *kernel_size) tensor
        Input tensor of patches
    dim : int
        Length of `kernel_size`.
    stride : [sequence of] int, default=`kernel_size`
        Stride.
    shape : sequence of int, optional
        Output shape. By default, it is computed from `spatial`,
        `stride` and `kernel_size`. If the output shape is larger than
        the computed shape, zero-padding is used.
        This parameter is mandatory if `collapsed = True`.
    collapsed : 'view' or bool, default=False
        Whether the spatial dimensions are collapsed in the input tensor.
        If 'view', use `view` instead of `reshape`, which will raise an
        error instead of triggering a copy when dimensions cannot be
        collapsed in a contiguous way.
    reduction : {'mean', 'sum', 'min', 'max'}, default='mean'
        Method to use to merge overlapping patches.

    Returns
    -------
    out : (..., *shape) tensor
        Folded tensor

    """
    def recon(x, stride):
        dim = len(stride)
        inshape = x.shape[-2*dim:-dim]
        batch_shape = x.shape[:-2*dim]
        indim = list(reversed(range(-1, -2 * dim - 1, -1)))
        outdim = (list(reversed(range(-2, -2 * dim - 1, -2))) +
                  list(reversed(range(-1, -2 * dim - 1, -2))))
        x = movedim(x, indim, outdim)
        outshape = [i * k for i, k in zip(inshape, stride)]
        x = x.reshape([*batch_shape, *outshape])
        return x

    inp = torch.as_tensor(inp)
    if torch.is_tensor(shape):
        shape = shape.tolist()
    dim = dim or (len(shape) if shape else None)
    if not dim:
        raise ValueError('Cannot guess dim from inputs')
    kernel_size = inp.shape[-dim:]
    stride = make_list(stride, len(kernel_size))
    stride = [st or sz for st, sz in zip(stride, kernel_size)]
    if any(sz > 2*st for st, sz in zip(stride, kernel_size)):
        # I only support overlapping of two patches (along a given dim).
        # If the kernel  is too large, more than two patches can overlap
        # and this function fails.
        raise ValueError('This function only works if kernel_size <= 2*stride')
    if not shape:
        if collapsed:
            raise ValueError('`shape` is mandatory when `collapsed=True`')
        inshape = inp.shape[-dim*2:-dim]
        shape = [(i-1)*st + sz
                 for i, st, sz in zip(inshape, stride, kernel_size)]
    else:
        inshape = [(o - sz) // st + 1
                   for o, st, sz in zip(shape, stride, kernel_size)]

    if collapsed:
        batch_shape = inp.shape[:-dim-1]
        inp = inp.reshape([*batch_shape, *inshape, *kernel_size])
    batch_shape = inp.shape[:-2*dim]

    # When the stride is equal to the kernel size, folding is easy
    # (it is obtained by shuffling dimensions and reshaping)
    # However, in the more general case, patches can overlap or,
    # conversely, have gaps between them. In the first case,
    # overlapping values must be reduced somehow. In the second case,
    # patches must be padded.

    # 1) padding (stride > kernel_size)
    padding = [max(0, st - sz) for st, sz in zip(stride, kernel_size)]
    padding = [0] * (inp.dim() - dim) + padding
    inp = pad(inp, padding, side='right')
    stride = [(st if st < sz else sz) for st, sz in zip(stride, kernel_size)]
    kernel_size = inp.shape[-dim:]

    # 2) merge overlaps
    overlap = [max(0, sz - st) for st, sz in zip(stride, kernel_size)]
    if any(o != 0 for o in overlap):
        slicer = [slice(None)] * (inp.dim() - dim)
        slicer += [slice(k) for k in stride]
        out = inp[tuple(slicer)].clone()
        if reduction == 'mean':
            count = inp.new_ones([*inshape, *stride], dtype=torch.int)
            fn = 'sum'
        else:
            count = None
            fn = reduction

        # ! a bit of padding to save the last values
        padding = [1 if o else 0 for o in overlap] + [0] * dim
        if count is not None:
            count = pad(count, padding, side='right')
        padding = [0] * (out.dim() - 2*dim) + padding
        value = (dtypes.dtype(inp.dtype).min if fn == 'max' else
                 dtypes.dtype(inp.dtype).max if fn == 'max' else 0)
        out = pad(out, padding, value=value, side='right')

        slicer1 = [slice(-1 if o else None) for o in overlap]
        slicer2 = [slice(None)] * dim
        slicer1 += [slice(st) for st in stride]
        slicer2 += [slice(st) for st in stride]

        import itertools
        overlaps = itertools.product(*[[0, 1] if o else [0] for o in overlap])
        for overlap in overlaps:
            front_slicer = list(slicer1)
            back_slicer = list(slicer2)
            for d, o in enumerate(overlap):
                if o == 0:
                    continue
                front_slicer[-dim+d] = slice(o)
                front_slicer[-2*dim+d] = slice(1, None)
                back_slicer[-dim+d] = slice(-o, None)
                back_slicer[-2*dim+d] = slice(None)
            if count is not None:
                count[tuple(front_slicer)] += 1
            front_slicer = (Ellipsis, *front_slicer)
            back_slicer = (Ellipsis, *back_slicer)

            if fn == 'sum':
                out[front_slicer] += inp[back_slicer]
            elif fn == 'max':
                out[front_slicer] = torch.max(out[front_slicer], inp[back_slicer])
            elif fn == 'min':
                out[front_slicer] = torch.min(out[front_slicer], inp[back_slicer])
            else:
                raise ValueError(f'Unknown reduction {reduction}')
        if count is not None:
            out /= count
    else:
        out = inp.clone()

    # end) reshape
    out = recon(out, stride)
    out = ensure_shape(out, [*batch_shape, *shape], side='right')

    return out


def histc(x, n=64, min=None, max=None, dim=None, keepdim=False, weights=None,
          order=1, bound='replicate', extrapolate=False, dtype=None):
    """Batched + differentiable histogram computation

    Parameters
    ----------
    x : tensor_like
        Input tensor.
    n : int, default=64
        Number of bins.
    min : float or tensor_like, optional
        Left edge of the histogram.
        Must be broadcastable to the input batch shape.
    max : float or tensor_like, optional
        Right edge of the histogram.
        Must be broadcastable to the input batch shape.
    dim : [sequence of] int, default=all
        Dimensions along which to compute the histogram
    keepdim : bool, default=False
        Keep singleton dimensions.
    weights : tensor, optional
        Observation weights
    order : {0..7}, default=1
        B-spline order encoding the histogram
    bound : bound_like, default='replicate'
        Boundary condition (only used when order > 1 or extrapolate is True)
    extrapolate : bool, default=False
        If False, discard data points that fall outside of [min, max]
        If True, use `bound` to assign them to a bin.
    dtype : torch.dtype, optional
        Output data type.
        Default: same as x unless it is not a floating point type, then
        `torch.get_default_dtype()`

    Returns
    -------
    h : (..., n) tensor
        Count histogram

    """
    # reshape as [batch, pool]]
    x = torch.as_tensor(x)
    if weights is not None:
        dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
        weights = torch.as_tensor(weights, dtype=dtype, device=x.device).expand(x.shape)
    if dim is None:
        x = x.reshape([1, -1])
        batch = []
        if weights is not None:
            weights = weights.reshape([1, -1])
    else:
        dim = py.make_list(dim)
        odim = list(range(-len(dim), 0))
        inshape = x.shape
        x = movedim(x, dim, odim)
        batch = x.shape[:-len(dim)]
        pool = x.shape[-len(dim):]
        x = x.reshape([-1, py.prod(pool)])
        if weights is not None:
            weights = weights.reshape([-1, py.prod(pool)])

    # compute limits
    if min is None:
        min = x.min(dim=-1, keepdim=True).values
    else:
        min = torch.as_tensor(min)
        min = min.expand(batch).reshape([-1, 1])
    if max is None:
        max = x.max(dim=-1, keepdim=True).values
    else:
        max = torch.as_tensor(max)
        max = max.expand(batch).reshape([-1, 1])

    # convert intensities to coordinates
    # (min -> -0.5  // max -> n-0.5)
    if not dtypes.dtype(x.dtype).is_floating_point:
        ftype = torch.get_default_dtype()
        x = x.to(ftype)
    x = x.clone()
    x = x.mul_(n / (max - min)).add_(n / (1 - max / min)).sub_(0.5)

    # push data into the histogram
    if not extrapolate:
        # hidden feature: tell pullpush to use +/- 0.5 tolerance when
        # deciding if a coordinate is inbounds.
        extrapolate = 2
    if weights is None:
        # count == push an image of ones
        h = GridCount.apply(x[:, :, None], [n], order, bound, extrapolate)[:, 0, ]
    else:
        # push weights
        h = GridPush.apply(weights[:, None, :], x[:, :, None], [n], order, bound, extrapolate)[:, 0, ]

    # reshape
    h = h.to(dtype)
    if keepdim:
        oshape = list(inshape)
        for d in dim:
            oshape[d] = 1
        oshape += [n]
    else:
        oshape = [*batch, n]
    h = h.reshape(oshape)
    return h


def histc2(x, n=64, min=None, max=None, dim=None, keepdim=False,
           order=1, bound='replicate', extrapolate=False, dtype=None):
    """Batched + differentiable joint histogram computation

    Parameters
    ----------
    x : (..., 2) tensor_like
        Input tensor.
    n : int or (int, int), default=64
        Number of bins.
    min : float or tensor_like, optional
        Left edge of the histogram.
        Must be broadcastable to (*batch, 2).
    max : float or tensor_like, optional
        Right edge of the histogram.
        Must be broadcastable to (*batch, 2).
    dim : [sequence of] int, default=all
        Dimensions along which to compute the histogram
    keepdim : bool, default=False
        Keep singleton dimensions.
    order : {0..7}, default=1
        B-spline order encoding the histogram
    bound : bound_like, default='replicate'
        Boundary condition (only used when order > 1 or extrapolate is True)
    extrapolate : bool, default=False
        If False, discard data points that fall outside of [min, max]
        If True, use `bound` to assign them to a bin.
    dtype : torch.dtype, optional
        Output data type.
        Default: same as x unless it is not a floating point type, then
        `torch.get_default_dtype()`

    Returns
    -------
    h : (..., n) tensor
        Count histogram

    """
    n = py.make_list(n, 2)

    # reshape as [batch, pool, 2]]
    x = torch.as_tensor(x)
    bck = backend(x)
    if dim is None:
        x = x.reshape([1, -1, 2])
        batch = []
    else:
        dim = py.make_list(dim)
        if -1 in dim or (x.dim()-1) in dim:
            raise ValueError('Cannot pool along last dimension')
        odim = list(range(-len(dim)-1, -1))
        inshape = x.shape
        x = movedim(x, dim, odim)
        batch = x.shape[:-len(dim)-1]
        pool = x.shape[-len(dim)-1:-1]
        x = x.reshape([-1, py.prod(pool), 2])

    # compute limits
    if min is None:
        min = x.detach().min(dim=-2, keepdim=True).values
    else:
        min = torch.as_tensor(min, **bck)
        min = min.expand([*batch, 2]).reshape([-1, 1, 2])
    if max is None:
        max = x.detach().max(dim=-2, keepdim=True).values
    else:
        max = torch.as_tensor(max, **bck)
        max = max.expand([*batch, 2]).reshape([-1, 1, 2])

    # convert intensities to coordinates
    # (min -> -0.5  // max -> n-0.5)
    if not dtypes.dtype(x.dtype).is_floating_point:
        ftype = torch.get_default_dtype()
        x = x.to(ftype)
    x = x.clone()
    nn = torch.as_tensor(n, dtype=x.dtype, device=x.device)
    x = x.mul_(nn / (max - min)).add_(nn / (1 - max / min)).sub_(0.5)

    # push data into the histogram
    if not extrapolate:
        # hidden feature: tell pullpush to use +/- 0.5 tolerance when
        # deciding if a coordinate is inbounds.
        extrapolate = 2
    h = GridCount.apply(x[:, None], n, order, bound, extrapolate)[:, 0]

    # reshape
    h = h.to(dtype)
    if keepdim:
        oshape = list(inshape)
        for d in dim:
            oshape[d] = 1
        oshape += n
    else:
        oshape = [*batch, *n]
    h = h.reshape(oshape)
    return h


def _hist_to_quantile(hist, q):
    """Compute quantiles from a cumulative histogram.

    Parameters
    ----------
    hist : (B, K) tensor
        Strictly monotonic cumulative histogram.
        B = batch size, K = number of bins
    q : (Q,) tensor
        Quantiles to compute.
        Q = number of quantiles.

    Returns
    -------
    values : (B, Q) tensor
        Quantile values, expressed in bins.
        They can be converted to values by `vmin + values * bin_width`.

    """
    hist, q = to_max_backend(hist, q, force_float=True)
    # compute the distance between discrete quantiles and target quantile
    hist = hist[:, None, :] - q[None, :, None]
    # find discrete quantile nearest to target quantile
    tmp = hist.clone()
    tmp[tmp < 0] = inf  # approach from below
    delta1, binq = tmp.min(dim=-1)
    # compute left weight (this is super ugly)
    delta0 = hist.neg().gather(-1, (binq - 1).clamp_min_(0)[..., None])[..., 0]
    delta0[binq == 0] = q.expand(delta0.shape)[binq == 0]
    del hist
    # compute interpolation weights
    delta0, delta1 = (delta1 / (delta0 + delta1), delta0 / (delta0 + delta1))
    # interpolate value
    q = delta0 * binq + delta1 * (binq + 1)
    return q


def quantile(input, q, dim=None, keepdim=False, bins=None, mask=None, *, out=None):
    """Compute quantiles.

    Parameters
    ----------
    input : tensor_like
        Input Tensor.
    q : float or (K,) tensor_like
        Values in [0, 1]: quantiles to computes
    dim : [sequence of] int, default=all
        Dimensions to reduce
    keepdim : bool, default=False
        Whether to squeeze reduced dimensions.
    bins : int, optional
        Number of histogram bins to use for fast quantile computation.
        By default: exact (but slow) computation using sorting.
    out : tensor, optional
        Output placeholder.

    Returns
    -------
    quant : (..., [K]) tensor
        Quantiles

    """
    def torch_is_recent():
        version = torch.__version__.split('.')
        version = (int(version[0]), int(version[1]))
        return version[0] > 2 or (version[0] == 1 and version[1] >= 7)

    input, q = to_max_backend(input, q)
    dim = py.make_list([] if dim is None else dim)
    # if torch_is_recent() and len(dim) < 2 and not bins:
    #     dim = dim[0] if dim else None
    #     return torch.quantile(input, q, dim=dim, keepdim=keepdim, out=out)

    # ------------------
    # our implementation
    # ------------------

    # reshape as (batch, pool)
    inshape = input.shape
    if mask is not None:
        mask = mask.expand(inshape)
    if not dim:
        if mask is not None:
            mask = mask.reshape([1, -1])
        input = input.reshape([1, -1])
        batch = []
    else:
        odim = list(range(-len(dim), 0))
        input = movedim(input, dim, odim)
        batch = input.shape[:-len(dim)]
        pool = input.shape[-len(dim):]
        input = input.reshape([-1, py.prod(pool)])
        if mask is not None:
            mask = movedim(mask, dim, odim).reshape([-1, py.prod(pool)])

    q_scalar = q.dim() == 0
    q = q.reshape([-1]).clone()
    if not bins and mask is None:
        # sort and sample
        input, _ = input.sort(-1)
        q = q.mul_(input.shape[-1]-1)
        q = GridPull.apply(input[None], q[None, :, None], 1, 'replicate', 0)[0]
    elif not bins:
        input, index = input.sort(-1)
        mask = mask[:, index]
        mask = mask.cumsum(-1) / mask.sum(-1, keepdim=True)
        mask[:, -1] = 1
        q = _hist_to_quantile(mask, q)
        q = GridPull.apply(input[None], q[None, :, None], 1, 'replicate', 0)[0]
    else:
        # compute cumulative histogram
        min = input.min(-1).values
        max = input.max(-1).values
        bin_width = (max-min)/bins
        hist = histc(input, bins, dim=-1, min=min, max=max, weights=mask)
        del max, input
        hist += eps(hist.dtype)  # ensures monotonicity
        hist = hist.cumsum(-1) / hist.sum(-1, keepdim=True)
        hist[..., -1] = 1  # avoid rounding errors
        # interpolate quantile value
        q = _hist_to_quantile(hist, q)
        q = min[:, None] + q * bin_width[:, None]

    # reshape
    if keepdim:
        if not dim:
            oshape = [1] * len(inshape)
        else:
            oshape = list(inshape)
            for d in dim:
                oshape[d] = 1
        oshape += [q.shape[-1]]
    else:
        oshape = [*batch, q.shape[-1]]
    q = q.reshape(oshape)
    if q_scalar:
        q = q.squeeze(-1)

    if out:
        out.reshape(q.shape).copy_(q)
    return q


@torch.jit.script
def _one_hot_wrapper(x: Tensor, dtype: Optional[torch.dtype] = None):
    x = x.long()
    x = torch.nn.functional.one_hot(x)
    x = x.to(dtype)
    return x


def one_hot(x, dim=-1, exclude_labels=None, exclude_missing=False, max_label=None,
            implicit=False, implicit_index=0, dtype=None, return_lookup=False):
    """One-hot encode a volume of labels.

    Parameters
    ----------
    x : tensor
        An integer-type tensor with label values.
    dim : int, default=-1
        Dimension in which to insert the one-hot channel.
    exclude_labels : sequence[int], optional
        A list of labels to exclude from one-hot encoding.
    exclude_missing : bool, default=False
        Exclude missing labels from one-hot encoding
        (their channel will be squeezed)
    max_label : int, optional
        Maximum label value
    implicit : bool, default=False
        Make the returned tensor have an implicit background class.
        In this case, output probabilities do not sum to one, but to some
        value smaller than one.
    implicit_index : int, default=-1
        Output channel to make implicit
    dtype : tensor.dtype, optional
        Output data type.
    return_lookup : bool, default=False
        Return lookup table from one-hot indices to labels

    Returns
    -------
    y : tensor
        One-hot tensor.
        The number of one-hot channels is equal to `x.max() - len(exclude) + 1`
        if not `implicit` else `x.max() - len(exclude)`.

    """
    if not exclude_labels and not exclude_missing and not implicit and not max_label:
        x = _one_hot_wrapper(x, dtype)
        x = fast_movedim(x, -1, dim)
        return x

    nb_classes = (max_label or int(x.max().item())) + 1
    exclude_labels = set(py.ensure_list(exclude_labels or []))
    if exclude_missing:
        all_labels = x.unique()
        missing_labels = [i for i in range(nb_classes) if i not in all_labels]
        exclude_labels = exclude_labels.union(missing_labels)

    dtype = dtype or x.dtype
    out = torch.zeros([nb_classes-implicit, *x.shape], dtype=dtype, device=x.device)
    implicit_index = (nb_classes + implicit_index if implicit_index < 0 else
                      implicit_index)
    i = 0
    lookup = []
    for j in range(nb_classes):
        if j in exclude_labels:
            continue
        if i == implicit_index:
            implicit_index = None
            continue
        out[i] = (x == j)
        lookup.append(j)
        i += 1

    out = fast_movedim(out, 0, dim)
    return (out, lookup) if return_lookup else out


def merge_labels(x, lookup):
    """Relabel a label tensor according to a lookup table

    Parameters
    ----------
    x : tensor
    lookup : sequence of [sequence of] int

    Returns
    -------
    x : tensor

    """
    out = torch.zeros_like(x)
    for i, j in enumerate(lookup):
        j = py.make_list(j)
        out[isin(x, j)] = i
    return out


class benchmark:
    """Context manager for the convolution benchmarking utility
    from pytorch.

    When the benchmark value is True, each time a convolution is called
    on a new input shape, several algorithms are performed and evaluated,
    and the best one kept in memory. Therefore, benchmarking is beneficial
    if and only if the (channel + spatial) shape of your input data is
    constant.

    Examples
    --------
    ```python
    from nitorch.core.utils import benchmark
    with benchmark(True):
        train_my_model(model)
    ```

    """

    def __init__(self, value=True):
        self.do_benchmark = value

    def __enter__(self):
        self.prev_value = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self.do_benchmark

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.backends.cudnn.benchmark = self.prev_value
