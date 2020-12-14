"""PyTorch utilities."""


import torch
import torch.nn.functional as F
from .pyutils import make_list, make_tuple
from .constants import inf
# from ._dtypes import astorch as dtype_astorch
from .dtypes import as_torch as dtype_astorch
import numbers


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
                dtype, device = info(*subs)
                subs = [elem.to(dtype=dtype, device=device) for elem in subs]
                return torch.stack(subs)
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
        # FIXME: does not work when inputs are lists
        d += extra_dims
        for _ in range(nd):
            input = torch.unsqueeze(input, min(d, input.dim()) if d > 0 else
                                           max(d, -(input.dim()+1)))
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


def info(*args):
    """Get the dtype and device of the first tensor of a list of objects."""
    for a in args:
        if torch.is_tensor(a):
            return a.dtype, a.device
    a = torch.as_tensor(args[0])
    return a.dtype, a.device


def to_common(*args, force_float=False):
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
    dtype = max_dtype(*args, force_float=force_float)
    device = max_device(*args)
    if len(args) == 1:
        return torch.as_tensor(args[0], dtype=dtype, device=device)
    else:
        return tuple(torch.as_tensor(arg, dtype=dtype, device=device)
                     for arg in args)


def to_common_device(*args):
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
        if is_dtype(x):
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

    .. warning::
        This function makes use of zero strides, so more than
        one output values can point to the same memory location.
        It is advised not too write in these tensors.

    """
    if 'shape' in kwargs:
        shape = kwargs['shape']
    else:
        *tensors, shape = tensors
    tensors = [torch.as_tensor(tensor) for tensor in tensors]

    def error(s0, s1):
        raise ValueError('Incompatible shapes for broadcasting: {} and {}.'
                         .format(s0, s1))

    # -------------
    # Compute shape
    # -------------

    # 1. nb dimensions
    nb_dim = len(shape)
    for tensor in tensors:
        nb_dim = max(nb_dim, len(tensor.shape))

    # 2. pad with singleton dimensions
    pad_size = nb_dim - len(shape)
    ones = [1] * pad_size
    if side == 'left':
        shape = [*ones, *shape]
    else:
        shape = [*shape, *ones]
    for i, tensor in enumerate(tensors):
        shape1 = tensor.shape
        pad_size = nb_dim - len(shape1)
        ones = [1] * pad_size
        if side == 'left':
            shape1 = [*ones, *shape1]
        else:
            shape1 = [*shape1, *ones]
        shape = [max(s0, s1) if s0 == 1 or s1 == 1 or s0 == s1
                 else error(s0, s1) for s0, s1 in zip(shape, shape1)]

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
    side : {'pre', 'post', 'both'}, defualt='post'
        Side to pad

    Returns
    -------
    out : tensor
        Padded tensor with shape `shape`

    """
    inp = torch.as_tensor(inp)
    shape = list(shape)
    shape = shape + [1] * max(0, inp.dim() - len(shape))
    if inp.dim() < len(shape):
        inp = inp.reshape(inp.shape + (1,) * max(0, len(shape) - inp.dim()))
    inshape = inp.shape
    shape = [inshape[d] if shape[d] is None else shape[d]
             for d in range(len(shape))]
    ndim = len(shape)

    # crop
    index = tuple(slice(min(shape[d], inshape[d])) for d in range(ndim))
    inp = inp.__getitem__(index)

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


_modifiers = {
    'circular': lambda x, i, n: x,
    'replicate': lambda x, i, n: x,
    'reflect': lambda x, i, n: x,
    'reflect1': lambda x, i, n: x,
    'reflect2': lambda x, i, n: x,
    }


def pad(inp, padsize, mode='constant', value=0, side=None):
    # type: (torch.Pad, tuple[int], str, any, str) -> torch.Tensor
    """Pad a tensor.

    This function is a bit more generic than torch's native pad, but probably
    a bit slower:
        . works with any input type
        . works with arbitrarily large padding size
        . crops the tensor for negative padding values
        . implements additional padding modes
    When used with defaults parameters (side=None), it behaves
    exactly like `torch.nn.functional.pad`

    Boundary modes are:
        . 'circular'
            -> corresponds to the boundary condition of an FFT
        . 'reflect' or 'reflect1'
            -> corresponds to the boundary condition of a DCT-I
        . 'reflect2'
            -> corresponds to the boundary condition of a DCT-II
        . 'replicate'
            -> replicates border values
        . 'constant'
            -> pads with a constant value (defaults to 0)

    Side modes are 'pre', 'post', 'both' or None. If side is not None,
    inp.dim() values (or less) should be provided. If side is None,
    twice as many values should be provided, indicating different padding sizes
    for the 'pre' and 'post' sides. If the number of padding values is less
    than the dimension of the input tensor, zeros are prepended.

    Args:
        inp (tensor): Input tensor.
        padsize (sequence): Amount of padding in each dimension.
        mode (string,optional): 'constant', 'replicate', 'reflect1',
            'reflect2', 'circular'.
            Defaults to 'constant'.
        value (optional): Value to pad with in mode 'constant'.
            Defaults to 0.
        side: Use padsize to pad on left side ('pre'), right side ('post') or
            both sides ('both'). If None, the padding side for the left and
            right sides should be provided in alternate order.
            Defaults to None.

    Returns:
        Padded tensor.

    """
    # Argument checking
    if mode not in tuple(_bounds.keys()) + ('constant',):
        raise ValueError('Padding mode should be one of {}. Got {}.'
                         .format(tuple(_bounds.keys()) + ('constant',), mode))
    if side == 'both':
        padpre = padsize
        padpost = padsize
    elif side == 'pre':
        padpre = padsize
        padpost = (0,) * len(padpre)
    elif side == 'post':
        padpost = padsize
        padpre = (0,) * len(padpost)
    else:
        if len(padsize) % 2:
            raise ValueError('Padding length must be divisible by 2')
        padpre = padsize[::2]
        padpost = padsize[1::2]
    padpre = (0,) * max(0, inp.dim()-len(padpre)) + padpre
    padpost = (0,) * max(0, inp.dim()-len(padpost)) + padpost
    if inp.dim() != len(padpre) or inp.dim() != len(padpost):
        raise ValueError('Padding length too large')

    padpre = torch.as_tensor(padpre)
    padpost = torch.as_tensor(padpost)

    # Pad
    if mode == 'constant':
        return _pad_constant(inp, padpre, padpost, value)
    else:
        bound = _bounds[mode]
        modifier = _modifiers[mode]
        return _pad_bound(inp, padpre, padpost, bound, modifier)


def _pad_constant(inp, padpre, padpost, value):
    # Uses torch.nn.functional.pad
    # Convert pre and post to a single list
    padpre = padpre.tolist()
    padpost = padpost.tolist()
    padding = padpre * 2
    padding[1::2] = padpost[::-1]
    padding[::2] = padpre[::-1]
    # Apply padding
    inp = F.pad(inp, padding, mode='constant', value=value)
    return inp


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


def ceil_pow(t, p=2.0, l=2.0):
    """Ceils each element in vector t to the
    closest n that satisfies: l*p**n.

    This function is useful, for example, to ensure an image's dimensions
    work well in an encoding/decoding architecture.

    Parameters
    ----------
    t : (d, ), tensor
    p : float, default=2.0
    l : float, default=2.0

    Returns
    ----------
    ct : (d, ), tensor

    """
    ct = t.clone()  # Do not modify in-place
    device = ct.device
    dtype0 = ct.dtype
    dtype = torch.float32
    dim = torch.as_tensor(ct, dtype=dtype, device=device)
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

class benchmark:
    """Context manager for the voncolution benchar;inking utility
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
