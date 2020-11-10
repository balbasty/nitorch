"""Helpers to validate inputs to layers"""


def dim(dim, *tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        shape = tensor.shape
        if len(shape) != dim + 2:
            raise ValueError('Expected tensor to have shape (B, C, *spatial)'
                             ' with len(spatial) == {} but found {}.'
                             .format(dim, shape))


def shape(tensor1, tensor2, dims=None, broadcast_ok=False):
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
