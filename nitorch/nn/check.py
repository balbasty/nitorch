"""Helpers to validate inputs to layers"""


def dim(dim, *tensors):
    """Check that all tensors have `dim + 2` dimensions.

    Parameters
    ----------
    dim : int
        Expected spatial dimension
    tensors : tensor or None
        Inputs tensors

    Raises
    ------
    ValueError
        If not `all(tensor.dim == dim+2)`. `None` inputs are discarded.

    """
    for tensor in tensors:
        if tensor is None:
            continue
        shape = tensor.shape
        if len(shape) != dim + 2:
            raise ValueError('Expected tensor to have shape (B, C, *spatial)'
                             ' with len(spatial) == {} but found {}.'
                             .format(dim, shape))


def shape(tensor1, tensor2, dims=None, broadcast_ok=False):
    """Check that the dimensions of two tensors are compatible.

    Parameters
    ----------
    tensor1 : tensor or None
        First input tensor
    tensor2 : tensor or None
        Second input tensor
    dims : int or sequence[int], optional
        Dimensions to check. By default, all dimensions are checked.
    broadcast_ok : bool, default=False
        If `True`, accept dimensions that are compatible for
        broadcasting (_i.e._, one of them is 1).

    Raises
    ------
    ValueError
        If `tensor1.dim != tensor2.dim` or dimensions listed in `dim`
        are not compatible.
        Dimensions are deemed compatible if they are equal or (when
        `broadcast_ok is True`) if one of them is 1.

    """
    if tensor1 is None or tensor2 is None:
        return
    if tensor1.dim() != tensor2.dim():
        raise ValueError("Number of dimensions not consistent: {} vs {}."
                         .format(tensor1.dim(), tensor2.dim()))
    shape1 = tensor1.shape
    shape2 = tensor2.shape
    if dims is None:
        dims = range(len(shape1))
    problems = []
    for d in dims:
        if shape1[d] != shape2[d]:
            if not broadcast_ok or (shape1[d] != 1 and shape2[d] == 1):
                problems.append(d)
    if problems:
        raise ValueError('Dimensions {} of `source` and `target` are '
                         'not compatible: {} vs {}'
                         .format(problems, shape1, shape2))
