"""Mathematical functions."""
# mikael.brudfors@gmail.com
# yael.balbastre@gmail.com

import torch
import math as pymath
from typing import List

from .optionals import custom_fwd, custom_bwd
from .constants import inf, ninf
from . import py, utils
from .version import torch_version

Tensor = torch.Tensor


def round(t, decimals=0):
    """Round a tensor to the given number of decimals.

    Parameters
    ----------
    t : tensor
        Input tensor.
    decimals : int, default=0
        Round to this decimal.

    Returns
    -------
    t : tensor
        Rounded tensor.
    """
    return torch.round(t * 10 ** decimals) / (10 ** decimals)


if torch_version('>=', (1, 8)):
    def floor_div(x, y):
        return torch.div(x, y, rounding_mode='floor')
else:
    def floor_div(x, y):
        return (x / y).floor_()


# ======================================================================
#
#                             REDUCTIONS
#
# ======================================================================
"""
Reductions
==========
This first section reimplements several reduction functions (sum, mean,
max...), with a more consistent API than the native pytorch one:
- all functions can reduce across multiple dimensions simultaneously
- min/max/median functions only return the reduced tensor by default
  (not the original indices of the returned elements).
  They have a specific argument `return_indices` to request these indices.
- all functions have an `omitnan` argument, or alternatively a `nan`
  version (e.g., `nansum`) where `omitnan=True` by default.
The typical API for all functions is:
```{python}
def fn(input, dim=None, keepdim=False, omitnan=False, inplace=False, out=None):
  \"\"\"
  input   : tensor, Input tensor
  dim     : int or sequence[int], Dimensions to reduce (default: all)
  keepdim : bool, Do not sequeeze reduced dimensions (default: False)
  omitnan : bool, Discard NaNs form the reduction (default: False)
  inplace : bool, Allow the input tensor to be modified (default: False)
                  (only useful in conjunction with `omitnan=True`)
  out     : tensor, Output placeholder
  \"\"\"
```
Reduction functions that pick a value from the input tensor (e.g., `max`)
have the additional argument:
```{python}
def fn(..., return_indices=False):
  \"\"\"
  return_indices : bool, Also return indices of the picked elements
  \"\"\"
```
"""


def _reduce_index(fn, input, dim=None, keepdim=False, omitnan=False,
                  inplace=False, return_indices=False, out=None,
                  nanfn=lambda x: x):
    """Multi-dimensional reduction for min/max/median.

    Signatures
    ----------
    fn(input) -> Tensor
    fn(input, dim) -> Tensor
    fn(input, dim, return_indices=True) -> (Tensor, Tensor)

    Parameters
    ----------
    fn : callable
        Reduction function
    input : tensor_like
        Input tensor.
    dim : int or sequence[int]
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    omitnan : bool, default=False
        Omit NaNs (else, the reduction of a NaN and any other value is NaN)
    inplace : bool, defualt=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : bool, defualt=False
        Return index of the min/max value on top if the value
    out : tensor or (tensor, tensor), optional
        Output placeholder
    nanfn : callable, optional
        Preprocessing function for removing nans

    Returns
    -------
    output : tensor
        Reduced tensor
    indices : (..., [len(dim)]) tensor
        Indices of the min/max/median values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    if omitnan:
        # If omitnan, we call a function that does the pre and post processing
        # of nans and give it a pointer to ourselves so that it can call us
        # back
        def self(inp):
            return _reduce_index(fn, inp, dim=dim, keepdim=keepdim, omitnan=False,
                                 inplace=False, return_indices=return_indices,
                                 out=out)
        return nanfn(self, input, inplace)

    input = torch.as_tensor(input)
    if dim is None:
        # If min across the entire tensor -> call torch
        return fn(input)

    # compute shapes
    scalar_dim = torch.as_tensor(dim).dim() == 0
    dim = [d if d >= 0 else input.dim() + d for d in py.make_list(dim)]
    shape = input.shape
    subshape = [s for d, s in enumerate(shape) if d not in dim]
    keptshape = [s if d not in dim else 1 for d, s in enumerate(shape)]
    redshape = [shape[d] for d in dim]
    input = utils.movedim(input, dim, -1)    # move reduced dim to the end
    input = input.reshape([*subshape, -1])   # collapse reduced dimensions

    # prepare placeholder
    out_val, out_ind = py.make_list(out, 2, default=None)
    if out_ind and len(dim) > 1:
        out_ind_tmp = input.new_empty(subshape, dtype=torch.long)
    else:
        out_ind_tmp = out_ind
    if out_val is not None and out_ind_tmp is None:
        out_ind_tmp = input.new_empty(subshape, dtype=torch.long)
    elif out_ind_tmp is not None and out_val is None:
        out_val = input.new_empty(subshape)
    out = (out_val, out_ind_tmp) if out_val is not None else None

    input, indices = fn(input, dim=-1, out=out)       # perform reduction

    if keepdim:
        input = input.reshape(keptshape)  # keep reduced singleton dimensions

    if return_indices:
        # convert to (i, j, k) indices
        indices = utils.ind2sub(indices, redshape, out=out_ind)
        indices = utils.movedim(indices, 0, -1)
        if keepdim:
            indices = indices.reshape([*keptshape, -1])
        if scalar_dim:
            indices = indices[..., 0]
        return input, indices

    return input


def max(input, dim=None, keepdim=False, omitnan=False, inplace=False,
        return_indices=False, out=None):
    """Multi-dimensional max reduction.

    Signatures
    ----------
    max(input) -> Tensor
    max(input, dim) -> Tensor
    max(input, dim, return_indices=True) -> (Tensor, Tensor)

    Notes
    -----
    .. This function cannot compute the maximum of two tensors, it only
       computes the maximum of one tensor (along a dimension).

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int or sequence[int]
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    omitnan : bool, default=False
        Omit NaNs (else, the reduction of a NaN and any other value is NaN)
    inplace : bool, defualt=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : bool, default=False
        Return index of the max value on top if the value
    out : tensor or (tensor, tensor), optional
        Output placeholder

    Returns
    -------
    output : tensor
        Reduced tensor
    indices : (..., [len(dim)]) tensor
        Indices of the max values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, omitnan=omitnan, inplace=inplace,
               return_indices=return_indices, out=out)
    return _reduce_index(torch.max, input, **opt, nanfn=_nanmax)


def min(input, dim=None, keepdim=False, omitnan=False, inplace=False,
        return_indices=False, out=None):
    """Multi-dimensional min reduction.

    Signatures
    ----------
    min(input) -> Tensor
    min(input, dim) -> Tensor
    min(input, dim, return_indices=True) -> (Tensor, Tensor)

    Notes
    -----
    .. This function cannot compute the minimum of two tensors, it only
       computes the minimum of one tensor (along a dimension).

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int or sequence[int]
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    omitnan : bool, default=False
        Omit NaNs (else, the reduction of a NaN and any other value is NaN)
    inplace : bool, defualt=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : bool, default=False
        Return index of the min value on top if the value
    out : tensor or (tensor, tensor), optional
        Output placeholder

    Returns
    -------
    output : tensor
        Reduced tensor
    indices : (..., [len(dim)]) tensor
        Indices of the min values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, omitnan=omitnan, inplace=inplace,
               return_indices=return_indices, out=out)
    return _reduce_index(torch.min, input, **opt, nanfn=_nanmin)


def _nanmax(fn, input, inplace=False):
    """Replace `nan`` with `-inf`"""
    input = torch.as_tensor(input)
    mask = torch.isnan(input)
    if inplace and not input.requires_grad:
        input[mask] = ninf
    else:
        val_ninf = torch.as_tensor(ninf, dtype=input.dtype, device=input.device)
        input = torch.where(mask, val_ninf, input)
    return fn(input)


def nanmax(input, dim=None, keepdim=False, inplace=False,
           return_indices=False, out=None):
    """Multi-dimensional max reduction, excluding NaNs.

    Signatures
    ----------
    nanmax(input) -> Tensor
    nanmax(input, dim) -> Tensor
    nanmax(input, dim, return_indices=True) -> (Tensor, Tensor)

    Notes
    -----
    .. This function cannot compute the minimum of two tensors, it only
       computes the minimum of one tensor (along a dimension).

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int or sequence[int]
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    inplace : bool, default=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : bool, default=False
        Return index of the max value on top if the value
    out : tensor or (tensor, tensor), optional
        Output placeholder

    Returns
    -------
    output : tensor
        Reduced tensor
    indices : (..., [len(dim)]) tensor
        Indices of the max values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, inplace=inplace,
               return_indices=return_indices, out=out)
    return max(input, **opt, omitnan=True)


def _nanmin(fn, input, inplace=False):
    """Replace `nan`` with `inf`"""
    input = torch.as_tensor(input)
    mask = torch.isnan(input)
    if inplace and not input.requires_grad:
        input[mask] = ninf
    else:
        val_inf = torch.as_tensor(inf, dtype=input.dtype, device=input.device)
        input = torch.where(mask, val_inf, input)
    return fn(input)


def nanmin(input, dim=None, keepdim=False, inplace=False,
           return_indices=False, out=None):
    """Multi-dimensional min reduction, excluding NaNs.

    Signatures
    ----------
    nanmin(input) -> Tensor
    nanmin(input, dim) -> Tensor
    nanmin(input, dim, return_indices=True) -> (Tensor, Tensor)

    Notes
    -----
    .. This function cannot compute the minimum of two tensors, it only
       computes the minimum of one tensor (along a dimension).

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int or sequence[int]
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    inplace : bool, default=False
        Allow modifying the input tensor in-place
        (Only useful if `omitnan == True`)
    return_indices : bool, default=False
        Return index of the min value on top if the value
    out : tensor or (tensor, tensor), optional
        Output placeholder

    Returns
    -------
    output : tensor
        Reduced tensor
    indices : (..., [len(dim)]) tensor
        Indices of the min values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, inplace=inplace,
               return_indices=return_indices, out=out)
    return min(input, **opt, omitnan=True)


def median(input, dim=None, keepdim=False, omitnan=None, inplace=None,
           return_indices=False, out=None):
    """Multi-dimensional median reduction.

    Signatures
    ----------
    median(input) -> Tensor
    median(input, dim) -> Tensor
    median(input, dim, return_indices=True) -> (Tensor, Tensor)

    Note
    ----
    .. This function omits NaNs by default

    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int or sequence[int]
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    return_indices : bool, default=False
        Return index of the median value on top if the value
    out : tensor or (tensor, tensor), optional
        Output placeholder

    Returns
    -------
    output : tensor
        Reduced tensor
    indices : (..., [len(dim)]) tensor
        Indices of the median values.
        If `dim` is a scalar, the last dimension is dropped.
    """
    opt = dict(dim=dim, keepdim=keepdim, return_indices=return_indices, out=out)
    return _reduce_index(torch.median, input, **opt)


def sum(input, *args, omitnan=False, inplace=False, **kwargs):
    """Compute the sum of a tensor.

    Parameters
    ----------
    input : tensor
        Input tensor.
    dim : int or list[int], optional
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    omitnan : bool, default=False
        Omit NaNs in the sum.
    inplace : bool, default=False
        Authorize working inplace.
    dtype : dtype, default=input.dtype
        Accumulator data type
    out : tensor, optional
        Output placeholder.

    Returns
    -------
    out : tensor
        Output tensor
    """
    if omitnan:
        return nansum(input, *args, inplace=inplace, **kwargs)
    else:
        return torch.sum(input, *args, **kwargs)


def nansum(input, *args, inplace=False, **kwargs):
    """Compute the sum of a tensor, excluding nans.

    Parameters
    ----------
    input : tensor
        Input tensor.
    dim : int or list[int], optional
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    inplace : bool, default=False
        Authorize working inplace.
    dtype : dtype, default=input.dtype
        Accumulator data type
    out : tensor, optional
        Output placeholder.

    Returns
    -------
    out : tensor
        Output tensor
    """
    input = torch.as_tensor(input)
    if not inplace:
        input = input.clone()
    mask = torch.isnan(input)
    if input.requires_grad and input.is_leaf:
        zero = torch.as_tensor(0, dtype=input.dtype, device=input.device)
        input = torch.where(mask, zero, input)
    else:
        input.masked_fill_(mask, 0)
    return torch.sum(input, *args, **kwargs)


def mean(input, *args, omitnan=False, inplace=False, **kwargs):
    """Compute the mean of a tensor.

    Parameters
    ----------
    input : tensor
        Input tensor.
    dim : int or list[int], optional
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    omitnan : bool, default=False
        Omit NaNs in the sum.
    inplace : bool, default=False
        Authorize working inplace.
    dtype : dtype, default=input.dtype
        Accumulator data type
    out : tensor, optional
        Output placeholder.

    Returns
    -------
    out : tensor
        Output tensor
    """
    if omitnan:
        return nanmean(input, *args, inplace=inplace, **kwargs)
    else:
        return torch.mean(input, *args, **kwargs)


def nanmean(input, *args, inplace=False, **kwargs):
    """Compute the mean of a tensor, excluding nans.

    Parameters
    ----------
    input : tensor
        Input tensor.
    dim : int or list[int], optional
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    inplace : bool, default=False
        Authorize working inplace.
    dtype : dtype, default=input.dtype
        Accumulator data type
    out : tensor, optional
        Output placeholder.

    Returns
    -------
    out : tensor
        Output tensor
    """
    input = torch.as_tensor(input)
    if not inplace:
        input = input.clone()
    mask = torch.isnan(input)
    if input.requires_grad and input.is_leaf:
        zero = torch.as_tensor(0, dtype=input.dtype, device=input.device)
        input = torch.where(mask, zero, input)
    else:
        input.masked_fill_(mask, 0)
    mask = mask.bitwise_not_()
    weights = mask.sum(*args, **kwargs).to(kwargs.get('dtype', input.dtype))
    return torch.sum(input, *args, **kwargs) / weights


def var(input, *args, omitnan=False, inplace=False, **kwargs):
    """Compute the variance of a tensor, excluding nans.

    Parameters
    ----------
    input : tensor
        Input tensor.
    dim : int or list[int], optional
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    unbiased : bool, default=True
        Whether to use the unbiased estimation or not.
    omitnan : bool, default=False
        Omit NaNs.
    inplace : bool, default=False
        Authorize working inplace.
    dtype : dtype, default=input.dtype
        Accumulator data type

    Returns
    -------
    out : tensor
        Output tensor
    """
    if omitnan:
        return nanvar(input, *args, inplace=inplace, **kwargs)
    else:
        return torch.var(input, *args, **kwargs)


def nanvar(input, *args, unbiased=True, inplace=False, **kwargs):
    """Compute the variance of a tensor, excluding nans.

    Parameters
    ----------
    input : tensor
        Input tensor.
    dim : int or list[int], optional
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    unbiased : bool, default=True
        Whether to use the unbiased estimation or not.
    inplace : bool, default=False
        Authorize working inplace.
    dtype : dtype, default=input.dtype
        Accumulator data type

    Returns
    -------
    out : tensor
        Output tensor
    """

    input = torch.as_tensor(input)
    requires_grad = input.requires_grad
    inplace = inplace and not (requires_grad and input.is_leaf)
    if not inplace:
        input = input.clone()
    mask = torch.isnan(input)
    input.masked_fill_(mask, 0)
    mask = mask.bitwise_not_()
    weights = mask.sum(*args, **kwargs).to(kwargs.get('dtype', input.dtype))
    mean = torch.sum(input, *args, **kwargs).div_(weights)
    input = input.square() if requires_grad else input.square_()
    var = torch.sum(input, *args, **kwargs).div_(weights)
    var -= mean
    if unbiased:
        weights /= (weights - 1)
        var *= weights
    return var


def std(input, *args, omitnan=False, inplace=False, **kwargs):
    """Compute the standard deviation of a tensor, excluding nans.

    Parameters
    ----------
    input : tensor
        Input tensor.
    dim : int or list[int], optional
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    unbiased : bool, default=True
        Whether to use the unbiased estimation or not.
    omitnan : bool, default=False
        Omit NaNs.
    inplace : bool, default=False
        Authorize working inplace.
    dtype : dtype, default=input.dtype
        Accumulator data type

    Returns
    -------
    out : tensor
        Output tensor
    """
    if omitnan:
        return nanstd(input, *args, inplace=inplace, **kwargs)
    else:
        return torch.std(input, *args, **kwargs)


def nanstd(input, *args, unbiased=True, inplace=False, **kwargs):
    """Compute the standard deviation of a tensor, excluding nans.

    Parameters
    ----------
    input : tensor
        Input tensor.
    dim : int or list[int], optional
        Dimensions to reduce.
    keepdim : bool, default=False
        Keep reduced dimensions.
    unbiased : bool, default=True
        Whether to use the unbiased estimation or not.
    inplace : bool, default=False
        Authorize working inplace.
    dtype : dtype, default=input.dtype
        Accumulator data type

    Returns
    -------
    out : tensor
        Output tensor
    """
    input = nanvar(input, *args, unbiased=unbiased, inplace=inplace, **kwargs)
    input = input.sqrt_() if not input.requires_grad else input.sqrt()
    return input


# ======================================================================
#
#                              SIMPLEX
#
# ======================================================================
"""
Simplex
=======
This section concerns functions that deal with data lying on the simplex,
i.e., probabilities. Specifically, we implement `softmax`, `log_softmax`,
`logsumexp` and `logit`. While most of these functions already exist in
PyTorch, we define more generic function that accept an "implicit" class.
This implicit class exists due to the constrained nature of discrete
probabilities, which must sum to one, meaning that their space ("the simplex")
has one less dimensions than the number of classes. Similarly, we can restrain
the logit (= log probability) space to be of dimension K-1 by forcing one of
the classes to have logits of arbitrary value (e.g., zero). This trick
makes functions like softmax invertible.
Note that in the 2-class case, it is extremely common to work in this
implicit setting by using the sigmoid function over a single logit instead
of the softmax function over two logits.
All functions below accept an argument `implicit` which takes either one
(boolean) value or a tuple of two (boolean) values. The first value
specifies if the input tensor has an explicit class while the second value
specified if the output tensor should have an implicit class.
Note that to minimize the memory footprint and numerical errors, most
backward passes are explicitely reimplemented (rather than relying on
autmatic diffentiation). This is because these function involve multiple
calls to `log` and `exp`, which must all store their input in order to
backpropagate, whereas a single tensor needs to be stored to backpropagate
through the entire softmax function.
"""


def _lse_fwd(input, dim=-1, keepdim=False, implicit=False):
    input = torch.as_tensor(input).clone()

    lse = input.max(dim=dim, keepdim=True)[0]
    if implicit:
        zero = input.new_zeros([])
        lse = torch.max(lse, zero)

    input = input.sub_(lse).exp_().sum(dim=dim, keepdim=True)
    if implicit:
        input += lse.neg().exp_()
    lse += input.log_()

    if not keepdim:
        lse = lse.squeeze(dim=dim)

    return lse


def _lse_bwd(input, output_grad, dim=-1, keepdim=False, implicit=False):
    input = _softmax_fwd(input, dim, implicit)
    if not keepdim:
        output_grad = output_grad.unsqueeze(dim)
    input *= output_grad
    return input


class _LSE(torch.autograd.Function):
    """Log-Sum-Exp with implicit class."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input, dim, keepdim, implicit):

        # Save precomputed components of the backward pass
        needs_grad = torch.is_tensor(input) and input.requires_grad
        if needs_grad:
            ctx.save_for_backward(input)
            ctx.args = {'dim': dim, 'implicit': implicit, 'keepdim': keepdim}

        return _lse_fwd(input, dim=dim, keepdim=keepdim, implicit=implicit)

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):

        input, = ctx.saved_tensors
        return _lse_bwd(input, output_grad,
                        dim=ctx.args['dim'],
                        keepdim=ctx.args['keepdim'],
                        implicit=ctx.args['implicit']), None, None, None


def logsumexp(input, dim=-1, keepdim=False, implicit=False):
    """Numerically stabilised log-sum-exp (lse).

    Parameters
    ----------
    input : tensor
        Input tensor.
    dim : int, default=-1
        The dimension or dimensions to reduce.
    keepdim : bool, default=False
        Whether the output tensor has dim retained or not.
    implicit : bool, default=False
        Assume that an additional (hidden) channel with value zero exists.

    Returns
    -------
    lse : tensor
        Output tensor.
    """
    return _LSE.apply(input, dim, keepdim, implicit)


def _add_class(x, bg, dim, index):
    # for implicit softmax
    if isinstance(bg, (int, float)):
        bg = torch.as_tensor(bg, **utils.backend(x))
        bgshape = list(x.shape)
        bgshape[dim] = 1
        bg = bg.expand(bgshape)
    if index in (-1, x.shape[dim]-1):
        pieces = [x, bg]
    elif index in (0, -dim):
        pieces = [bg, x]
    else:
        pieces = [
            utils.slice_tensor(x, slice(index), dim),
            bg,
            utils.slice_tensor(x, slice(index, None), dim)]
    return torch.cat(pieces, dim=dim)


def _remove_class(x, dim, index):
    # for implicit softmax
    if index in (-1, x.shape[dim]-1):
        x = utils.slice_tensor(x, slice(-1), dim)
    elif index in (0, -dim):
        x = utils.slice_tensor(x, slice(1, None), dim)
    else:
        x = torch.cat([
            utils.slice_tensor(x, slice(index), dim),
            utils.slice_tensor(x, slice(index+1, None), dim)])
    return x


def _softmax_fwd(input, dim=-1, implicit=False, implicit_index=0, inplace=False):
    implicit_in, implicit_out = py.ensure_list(implicit, 2)

    maxval, _ = torch.max(input, dim=dim, keepdim=True)
    if implicit_in:
        maxval.clamp_min_(0)  # don't forget the class full of zeros

    if not inplace:
        input = input.clone()

    input = input.sub_(maxval).exp_()
    sumval = torch.sum(input, dim=dim, keepdim=True,
                       out=maxval if not implicit_in else None)
    if implicit_in:
        sumval += maxval.neg().exp_()  # don't forget the class full of zeros
    input /= sumval

    if implicit_in and not implicit_out:
        background = input.sum(dim, keepdim=True).neg_().add_(1)
        input = _add_class(input, background, dim, implicit_index)
    elif implicit_out and not implicit_in:
        input = _remove_class(input, dim, implicit_index)

    return input


def _softmax_bwd(output, output_grad, dim=-1, implicit=False, implicit_index=0):
    implicit = py.ensure_list(implicit, 2)
    add_dim = implicit[1] and not implicit[0]
    drop_dim = implicit[0] and not implicit[1]

    grad = output_grad.clone()
    del output_grad
    grad *= output
    gradsum = grad.sum(dim=dim, keepdim=True)
    grad = grad.addcmul_(gradsum, output, value=-1)  # grad -= gradsum * output
    if add_dim:
        grad_background = output.sum(dim=dim, keepdim=True).neg().add(1)
        grad_background.mul_(gradsum.neg_())
        grad = _add_class(grad, grad_background, dim, implicit_index)
    elif drop_dim:
        grad = utils.slice_tensor(grad, slice(-1), dim)

    return grad


class _Softmax(torch.autograd.Function):
    """Softmax with implicit class."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input, dim, implicit, implicit_index, inplace):

        # Save precomputed components of the backward pass
        needs_grad = torch.is_tensor(input) and input.requires_grad
        # Compute matrix exponential
        s = _softmax_fwd(input, dim=dim, implicit=implicit,
                         implicit_index=implicit_index, inplace=inplace)

        if needs_grad:
            ctx.save_for_backward(s)
            ctx.args = {'dim': dim, 'implicit': implicit}

        return s

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        s, = ctx.saved_tensors
        out = _softmax_bwd(s, output_grad,  dim=ctx.args['dim'], implicit=ctx.args['implicit'])
        return out, None, None, None, None


def logit(input, dim=-1, implicit=False, implicit_index=0, inplace=False):
    """(Multiclass) logit function

    Notes
    -----
    .. logit(x)_k = log(x_k) - log(x_K), where K is an arbitrary channel.
    .. The logit function is the inverse of the softmax function:
        .. logit(softmax(x, implicit=True), implicit=True) == x
        .. softmax(logit(x, implicit=True), implicit=True) == x
    .. Note that when `implicit=False`, softmax is surjective (many
       possible logits map to the same simplex value). We only have:
        .. softmax(logit(x, implicit=False), implicit=False) == x
    .. `logit(x, implicit=True)`, with `x.shape[dim] == 1` is equivalent
       to the "classical" binary logit function (inverse of the sigmoid).

    Parameters
    ----------
    input : tensor
        Tensor of probabilities.
    dim : int, default=-1
        Simplex dimension, along which the logit is performed.
    implicit : bool or (bool, bool), default=False
        The first value relates to the input tensor and the second
        relates to the output tensor.
        - implicit[0] == True assumes that an additional (hidden) channel
          exists, such as the sum along `dim` is one.
        - implicit[1] == True drops the implicit channel from the
          logit tensor.
    implicit_index : int, default=0
        Index of the implicit channel. This is the channel whose logits
        are assumed equal to zero.
    inplace : bool, default=False
        Perform conversion in-place if possible.

    Returns
    -------
    output : tensor
    """
    implicit = py.ensure_list(implicit, 2)
    if implicit[0]:
        input_extra = input.sum(dim).neg_().add_(1).clamp_min_(1e-8).log_()
        input = input.log()
    else:
        input = input.log()
        input_extra = utils.slice_tensor(input, implicit_index, dim)
        if implicit[1]:
            input = _remove_class(input, dim, implicit_index)
    input_extra = input_extra.unsqueeze(dim)
    input -= input_extra.clone()
    if implicit[0] and not implicit[1]:
        input = _add_class(input, 0, dim, implicit_index)
    return input


def softmax(input, dim=-1, implicit=False, implicit_index=0, inplace=False):
    """ SoftMax (safe).

    Parameters
    ----------
    input : tensor
        Tensor with values.
    dim : int, default=-1
        Dimension to take softmax, defaults to last dimensions.
    implicit : bool or (bool, bool), default=False
        The first value relates to the input tensor and the second
        relates to the output tensor.
        - implicit[0] == True assumes that an additional (hidden) channel
          with value zero exists.
        - implicit[1] == True drops the last class from the
          softmaxed tensor.
    implicit_index : int, default=0
    inplace : bool, default=False
        Apply function in-place, if possible.

    Returns
    -------
    output : tensor
        Soft-maxed tensor with values.
    """
    input = torch.as_tensor(input)
    return _Softmax.apply(input, dim, implicit, implicit_index, inplace)


def log_softmax(input, dim=-1, implicit=False, implicit_index=0):
    """ Log(SoftMax).

    Parameters
    ----------
    input : tensor
        Tensor with values.
    dim : int, default=-1
        Dimension to take softmax, defaults to last dimensions.
    implicit : bool or (bool, bool), default=False
        The first value relates to the input tensor and the second
        relates to the output tensor.
        - implicit[0] == True assumes that an additional (hidden) channel
          with value zero exists.
        - implicit[1] == True drops the last class from the
          softmaxed tensor.
    implicit_index : int, default=0

    Returns
    -------
    output : tensor
        Log-Soft-maxed tensor with values.
    """
    input = torch.as_tensor(input)
    implicit = py.ensure_list(implicit, 2)
    lse = logsumexp(input, dim=dim, implicit=implicit[0], keepdim=True)
    if implicit[0] and not implicit[1]:
        output = _add_class(input, 0, dim, implicit_index)
        output -= lse
    elif implicit[1] and not implicit[0]:
        input = _remove_class(input, dim, implicit_index)
        output = input - lse
    else:
        output = input - lse
    return output


def softmax_lse(input, dim=-1, lse=False, weights=None, implicit=False,):
    """ SoftMax (safe).

    Parameters
    ----------
    input : torch.tensor
        Tensor with values.
    dim : int, default=-1
        Dimension to take softmax, defaults to last dimensions.
    lse : bool, default=False
        Compute log-sum-exp as well.
    weights : torch.tensor, optional:
        Observation weights (only used in the log-sum-exp).
    implicit : bool or (bool, bool), default=False
        The first value relates to the input tensor and the second
        relates to the output tensor.
        - implicit[0] == True assumes that an additional (hidden) channel
          with value zero exists.
        - implicit[1] == True drops the last class from the
          softmaxed tensor.

    Returns
    -------
    Z : torch.tensor
        Soft-maxed tensor with values.
    """
    def sumto(x, *a, out=None, **k):
        if out is None or x.requires_grad:
            return torch.sum(x, *a, **k)
        else:
            return torch.sum(x, *a, **k, out=out)

    implicit_in, implicit_out = py.make_list(implicit, 2)

    maxval, _ = torch.max(input, dim=dim, keepdim=True)
    if implicit_in:
        maxval.clamp_min_(0)  # don't forget the class full of zeros

    input = (input-maxval).exp()
    sumval = sumto(input, dim=dim, keepdim=True,
                   out=maxval if not lse else None)
    if implicit_in:
        sumval += maxval.neg().exp()  # don't forget the class full of zeros
    input = input / sumval

    if lse:
        # Compute log-sum-exp
        #   maxval = max(logit)
        #   lse = maxval + log[sum(exp(logit - maxval))]
        # If implicit
        #   maxval = max(max(logit),0)
        #   lse = maxval + log[sum(exp(logit - maxval)) + exp(-maxval)]
        sumval = sumval.log()
        maxval += sumval
        if weights is not None:
            maxval *= weights
        maxval = maxval.sum(dtype=torch.float64)
    else:
        maxval = None

    if implicit_in and not implicit_out:
        background = input.sum(dim, keepdim=True).neg().add(1)
        input = torch.cat((input, background), dim=dim)
    elif implicit_out and not implicit_in:
        input = utils.slice_tensor(input, slice(-1), dim)

    if lse:
        return input, maxval
    else:
        return input


def softsort(x, tau=1, hard=False, descending=False, apply=False, pow=1):
    """SoftSort of a vector

    Parameters
    ----------
    x : (..., K) tensor
    tau : float, default=1
        Temperature
    hard : bool, default=False
        Perform (hard) sort based on the softsort
        (similar to applying argmax to a softmax)
    descending : bool, default=False
        Sort in descending order
    apply : bool, default=False
        Return the soft-sorted vector insteas of the sort-sorting weights
    pow : float, default=1
        Power of the norm

    Returns
    -------
    P : (..., K, K) tensor /or/ x : (..., K) if `apply`
        Soft sorting weights, normalized across the last dimension.
        /or/
        Sort-sorted input vector: `matvec(P, x)`

    References
    ----------
    .. [1] "SoftSort: A Continuous Relaxation for the argsort Operator"
           Sebastian Prillo & Julian Martin Eisenschlos
           ICML (2020)
           https://arxiv.org/abs/2006.16038
           https://github.com/sprillo/softsort (MIT license)

    """
    norm = ((lambda x: x.abs_()) if pow == 1 else
            (lambda x: x.square()) if pow == 2 else
            (lambda x: x.abs_().pow(pow)))

    x = x.unsqueeze(-1)
    sorted = x.sort(descending=True, dim=-2).values
    pairwise_diff = norm(sorted - x.transpose(-1, -2)).div_(-tau)
    P_hat = softmax(pairwise_diff, -1)

    if hard:
        P = torch.zeros_like(P_hat)
        P.scatter_(-1, torch.topk(P_hat, 1, -1).values, value=1)
        P_hat += P.sub_(P_hat).detach()
    if descending:
        P_hat = P_hat.flip(-2)
    if apply:
        return P_hat.matmul(x).squeeze(-1)
    return P_hat


def neuralsort(x, tau=1, hard=False, descending=False, apply=False):
    """SoftSort of a vector

    Parameters
    ----------
    x : (..., K) tensor
    tau : float, default=1
        Temperature
    hard : bool, default=False
        Perform (hard) sort based on the softsort
        (similar to applying argmax to a softmax)
    descending : bool, default=False
        Sort in descending order
    apply : bool, default=False
        Return the soft-sorted vector insteas of the sort-sorting weights

    Returns
    -------
    P : (..., K, K) tensor /or/ x : (..., K) if `apply`
        Soft sorting weights, normalized across the last dimension.
        /or/
        Sort-sorted input vector: `matvec(P, x)`

    References
    ----------
    .. [1] "Stochastic Optimization of Sorting Networks via Continuous Relaxations"
           Aditya Grover, Eric Wang, Aaron Zweig & Stefano Ermon
           ICLR (2019)
           https://arxiv.org/abs/1903.08850
           https://github.com/ermongroup/neuralsort (MIT license)

    """
    dim = x.shape[-1]
    dims = torch.arange(dim, **utils.backend(x))
    x = x.unsqueeze(-1)
    A = (x - x.transpose(-1, -2)).abs_()
    B = A.sum(-1, keepdim=True)
    scaling = dims.add_(1).mul_(-2).add_(dim+1)
    C = x * scaling

    P_max = C.sub_(B).transpose(-1, -2).div_(tau)
    P_hat = softmax(P_max, -1)

    if hard:
        P = torch.zeros_like(P_hat)
        P.scatter_(-1, torch.topk(P_hat, 1, -1).values, value=1)
        P_hat += P.sub_(P_hat).detach()
    if descending:
        P_hat = P_hat.flip(-2)
    if apply:
        return P_hat.matmul(x).squeeze(-1)
    return P_hat


# ======================================================================
#
#                           SPECIAL FUNCTIONS
#
# ======================================================================


if hasattr(torch, 'digamma'):
    @torch.jit.script
    def mvdigamma(input, order: int = 1):
        """Derivative of the log of the Gamma function, eventually multivariate
        Parameters
        ----------
        input : tensor
        order : int, default=1
        Returns
        -------
        tensor
        """
        dg = torch.digamma(input)
        for p in range(2, order + 1):
            dg += torch.digamma(input + (1 - p) / 2)
        return dg


# Bessel functions were written by John Ashburner <john@fil.ion.ucl.ac.uk>
# (Wellcome Centre for Human Neuroimaging, UCL, London, UK)


def besseli(nu, z, mode=None):
    """Modified Bessel function of the first kind
    Parameters
    ----------
    nu : float
    z  : tensor
    mode : {0 or None, 1 or 'norm', 2 or 'log'}
    Returns
    -------
    b : tensor
        besseli(nu,z)        if mode is None
        besseli(ni,z)/exp(z) if mode == 'norm'
        log(besseli(nu,z))   if mode == 'log'
    References
    ----------
    ..[1] "On the use of the noncentral chi-square density function
           for the distribution of helicopter spectral estimates."
          Garber DP.
          NASA, Langley Research Center (1993)
    """
    z = torch.as_tensor(z)
    is_scalar = z.dim() == 0
    if is_scalar:
        z = z[None]
    if not isinstance(mode, int):
        code = (2 if mode == 'log' else 1 if mode == 'norm' else 0)
    else:
        code = mode
    if nu == 0:
        z = besseli0(z, code)
    elif nu == 1:
        z = besseli1(z, code)
    else:
        z = besseli_any(nu, z, code)
    if is_scalar:
        z = z[0]
    return z



@torch.jit.script
def besseli0(z, code: int = 0):
    """Modified Bessel function of the first kind
    Parameters
    ----------
    z : tensor
    code : {0, 1, 2}
    Returns
    -------
    b : tensor
        besseli(0,z)        if code==0
        besseli(0,z)/exp(z) if code==1  ('norm')
        log(besseli(0,z))   if code==2  ('log')
    """
    f = torch.zeros_like(z)
    msk = z < 15.0/4.0

    # --- branch 1 ---
    if msk.any():
        zm = z[msk]
        t = (zm*(4.0/15.0))**2
        t = 1 + t*(3.5156229+t*(3.0899424+t*(1.2067492+t*(0.2659732+t*(0.0360768+t*0.0045813)))))
        if code == 2:
            f[msk] = t.log()
        else:
            if code == 1:
                t = t / zm.exp()
            f[msk] = t

    # --- branch 2 ---
    msk.bitwise_not_()
    if msk.any():
        zm = z[msk]
        t = (15.0/4.0)/zm
        t = (0.39894228+t*(0.01328592+t*(0.00225319+t*(-0.00157565+t*(0.00916281+t*(-0.02057706+t*(0.02635537+t*(-0.01647633+t*0.0039237))))))))
        t.clamp_min_(1e-32)
        if code == 2:
            f[msk] = zm - 0.5 * zm.log() + t.log()
        elif code == 1:
            f[msk] = t / zm.sqrt()
        else:
            f[msk] = zm.exp() * t / zm.sqrt()

    return f


@torch.jit.script
def besseli1(z, code: int = 0):
    """Modified Bessel function of the first kind
    Parameters
    ----------
    z : tensor
    code : {0, 1, 2}
    Returns
    -------
    b : tensor
        besseli(1,z)        if code==0
        besseli(1,z)/exp(z) if code==1  ('norm')
        log(besseli(1,z))   if code==2  ('log')
    """
    f = torch.zeros_like(z)
    msk = z < 15.0/4.0

    # --- branch 1 ---
    if msk.any():
        zm = z[msk]
        t = (zm*(4.0/15.0))**2
        t = 0.5+t*(0.87890594+t*(0.51498869+t*(0.15084934+t*(0.02658733+t*(0.00301532+t*0.00032411)))))
        if code == 2:
            f[msk] = zm.log() + t.log()
        elif code == 0:
            f[msk] = zm * t
        else:
            f[msk] = zm * t / zm.exp()

    # --- branch 2 ---
    msk.bitwise_not_()
    if msk.any():
        zm = z[msk]
        t = (15.0/4.0)/zm
        t = 0.398942281+t*(-0.03988024+t*(-0.00362018+t*(0.00163801+t*(-0.01031555+t*(0.02282967+t*(-0.02895312+t*(0.01787654-t*0.00420059)))))))
        if code == 2:
            f[msk] = zm - 0.5 * zm.log() + t.log()
        elif code == 0:
            f[msk] = zm.exp() * t / zm.sqrt()
        else:
            f[msk] = t / zm.sqrt()
    return f


@torch.jit.script
def besseli_small(nu: float, z, M: int = 64, code: int = 0):
    """Modified Bessel function of the first kind - series computation
    Parameters
    ----------
    nu : float
    z  : tensor
    M  : int, series length (bigger is more accurate, but slower)
    code : {0, 1, 2}
    Returns
    -------
    b : tensor
        besseli(nu,z)        if code==0
        besseli(nu,z)/exp(z) if code==1 ('norm')
        log(besseli(nu,z))   if code==2 ('log')
    """
    # The previous implementation of this function (`besseli_small_old`)
    # used `z` as a stabilizing pivot in the log-sum-exp. However,
    # this lead to underflows (`exp` returned zero). Instead, this new
    # implementation uses the first term in the sum (m=0) as pivot.
    # We therefore start with the second term and add 1 (= exp(0)) at the end.

    # NOTE: lgamma(3) = 0.693147180559945
    lgamma_nu_1 = pymath.lgamma(nu + 1)
    M = max(M, 2)
    x = torch.log(0.5*z)
    f = torch.exp(x * 2 - (0.693147180559945 + pymath.lgamma(nu + 2) - lgamma_nu_1))
    for m in range(2, M):
        f = f + torch.exp(x * (2*m) - (pymath.lgamma(m + 1) + pymath.lgamma(m + 1 + nu) - lgamma_nu_1))
    f = f + 1

    if code == 2:
        return f.log() + x * nu - lgamma_nu_1
    elif code == 1:
        return f * (x * nu - lgamma_nu_1 - z).exp()
    else:
        return f * (x * nu - lgamma_nu_1).exp()


# DEPRECATED
@torch.jit.script
def besseli_small_old(nu: float, z, M: int = 64, code: int = 0):
    """Modified Bessel function of the first kind - series computation
    Parameters
    ----------
    nu : float
    z  : tensor
    M  : int, series length (bigger is more accurate, but slower)
    code : {0, 1, 2}
    Returns
    -------
    b : tensor
        besseli(nu,z)        if code==0
        besseli(nu,z)/exp(z) if code==1 ('norm')
        log(besseli(nu,z))   if code==2 ('log')
    """
    M = max(M, 2)
    x = torch.log(0.5*z)
    f = torch.exp(x * nu - pymath.lgamma(nu + 1.0) - z)
    for m in range(1, M):
        f = f + torch.exp(x * (2.0*m + nu) - (pymath.lgamma(m+1.0) + pymath.lgamma(m+1.0 + nu)) - z)

    if code == 2:
        return f.log() + z
    elif code == 1:
        return f
    else:
        return f * z.exp()


@torch.jit.script
def besseli_large(nu: float, z, code: int = 0):
    """Modified Bessel function of the first kind
    Uniform asymptotic approximation (Abramowitz and Stegun p 378)
    Parameters
    ----------
    nu   : scalar float
    z    : torch tensor
    code : 0, 1 or 2
    Returns
    -------
    b : tensor
        besseli(nu,z)        if code==0
        besseli(nu,z)/exp(z) if code==1 ('norm')
        log(besseli(nu,z))   if code==2 ('log')
    """

    f = z/nu
    f = f*f
    msk = f > 4.0
    t = torch.zeros_like(f)

    # -- branch 1 ---
    if msk.any():
        tmp = torch.sqrt(1.0+f[msk].reciprocal())
        t[msk] = z[msk]*tmp/nu
        f[msk] = nu*(t[msk] + torch.log((nu/z[msk]+tmp).reciprocal()))
    # -- branch 2 ---
    msk = msk.bitwise_not_()
    if msk.any():
        tmp = torch.sqrt(1.0+f[msk])
        t[msk] = tmp.clamp_max(1)
        f[msk] = nu*(t[msk] + torch.log(z[msk]/(nu*(1.0+tmp))))

    t = t.reciprocal()
    tt = t*t
    ttt = t*tt
    us = 1.0
    den = nu
    us = us + t*(0.125 - tt*0.2083333333333333)/den
    den = den*nu
    us = us + tt*(0.0703125 + tt*(-0.4010416666666667 + tt*0.3342013888888889))/den
    den = den*nu
    us = us + ttt*(0.0732421875 + tt*(-0.8912109375 + tt*(1.846462673611111 - tt*1.025812596450617)))/den
    den = den*nu
    us = us + tt*tt*(0.112152099609375 + tt*(-2.3640869140625 + tt*(8.78912353515625 +
               tt*(-11.20700261622299 + tt*4.669584423426248))))/den
    den = den*nu
    us = us + tt*ttt*(0.2271080017089844 + tt*(-7.368794359479632 + tt*(42.53499874638846 +
               tt*(-91.81824154324002 + tt*(84.63621767460074 - tt*28.21207255820025)))))/den
    den = den*nu
    us = us + ttt*ttt*(0.5725014209747314 + tt*(-26.49143048695155 + tt*(218.1905117442116 +
               tt*(-699.5796273761326 + tt*(1059.990452528 + tt*(-765.2524681411817 +
               tt*212.5701300392171))))))/den

    if code == 2:
        f = f + 0.5*(torch.log(t)-pymath.log(nu)) - 0.918938533204673 # 0.5*log(2*pi)
        f = f + torch.log(us)
    elif code == 0:
        f = torch.exp(f)*torch.sqrt(t)*us*(0.398942280401433/pymath.sqrt(nu))
    else:
        f = torch.exp(f-z)*torch.sqrt(t)*us*(0.398942280401433/pymath.sqrt(nu))
    return f


@torch.jit.script
def besseli_any(nu: float, z, code: int = 0):
    """
    Modified Bessel function of the first kind
    Parameters
    ----------
    nu : float
    z  : tensor
    code : {0, 1, 2}
    Returns
    -------
    b : tensor
        besseli(nu,z)        if code==0
        besseli(nu,z)/exp(z) if code==1 ('norm')
        log(besseli(nu,z))   if code==2 ('log')
    """

    if nu >= 15.0:
        f = besseli_large(nu, z, code)
    else:
        thr = 5.0 * pymath.sqrt(15.0 - nu) * pymath.sqrt(nu + 15.0) / 3.0
        msk = z < 2.0 * thr
        f = torch.zeros_like(z)
        if msk.any():
            f[msk] = besseli_small(nu, z[msk], int(pymath.ceil(thr*1.9+2.0)), code)
        msk = msk.bitwise_not_()
        if msk.any():
            f[msk] = besseli_large(nu, z[msk], code)
    return f


@torch.jit.script
def besseli_ratio(nu: float, X, N: int = 4, K: int = 10):
    """Approximates ratio of the modified Bessel functions of the first kind:
       besseli(nu+1,x)/besseli(nu,x)
    Parameters
    ----------
    nu : float
    X : tensor
    N, K :  int
        Terms in summation, higher number, better approximation.
    Returns
    -------
    I : tensor
        Ratio of the modified Bessel functions of the first kind.
    References
    ----------
    ..[1] Amos DE. "Computation of modified Bessel functions and their ratios."
          Mathematics of Computation. 1974;28(125):239-251.
    """

    # Begin by computing besseli(nu+1+N,x)/besseli(nu+N,x)

    nu1 = nu + K
    rk: List[torch.Tensor] = []
    XX = X*X

    # Lower bound (eq. 20a)
    # for k=0:N, rk{k+1} = x./((nu1+k+0.5)+sqrt((nu1+k+1.5).^2+x.^2)); end
    for k in range(0, N + 1):
        tmp = XX + (nu1 + k + 1.5) ** 2
        tmp = tmp.sqrt()
        tmp = tmp + (nu1 + k + 0.5)
        tmp = X / tmp
        rk.append(tmp)

    for m in range(N, 0, -1):
        # Recursive updates (eq. 20b)
        # rk{k} = x./(nu1+k+sqrt((nu1+k).^2+((rk{k+1})./rk{k}).
        for k2 in range(1, m + 1):
            tmp = rk[k2] / rk[k2-1]
            tmp = tmp * XX
            tmp = tmp + (nu1 + k2) ** 2
            tmp = tmp.sqrt()
            tmp = tmp + (nu1 + k2)
            tmp = X / tmp
            rk[k2-1] = tmp
        rk.pop(-1)
    result = rk.pop(0)

    # Convert the result to besseli(nu+1,x)/besseli(nu,x) with
    # backward recursion (eq. 2).
    iX = X.reciprocal()
    for k3 in range(K, 0, -1):
        # result = 1. / (2. * (nu + k3) / X + result)
        result = result.add(iX, alpha=2 * (nu + k3))
        result = result.reciprocal()

    return result


# TODO:
#   The following functions should be replaced by tensor-compatible
#   equivalents in linalg


from numpy import real
from scipy.linalg import expm as expm_scipy
from scipy.linalg import logm as logm_scipy


def expm(M):
    """ Computes the matrix exponential of M.
    Args:
        M (torch.tensor): Square matrix (N, N)
    Returns:
        M (torch.tensor): Matrix exponential (N, N)
    """
    device = M.device
    dtype = M.dtype
    M = M.detach().cpu().numpy()
    M = expm_scipy(M)
    M = torch.from_numpy(M).type(dtype).to(device)
    return M


def logm(M):
    """ Computes the real matrix logarithm of M.
    Args:
        M (torch.tensor): Square matrix (N, N)
    Returns:
        M (torch.tensor): Matrix logarithm (N, N)
    """
    device = M.device
    dtype = M.dtype
    M = M.detach().cpu().numpy()
    M = logm_scipy(M)
    M = real(M)
    M = torch.from_numpy(M).type(dtype).to(device)
    return M


def besseli_old(X, order=0, Nk=64):
    """ Approximates the modified Bessel function of the first kind,
        of either order zero or one.
        OBS: Inputing float32 can lead to numerical issues.
    Args:
        X (torch.tensor): Input (N, 1).
        order (int, optional): 0 or 1, defaults to 0.
        Nk (int, optional): Terms in summation, higher number, better approximation.
            Defaults to 50.
    Returns:
        I (torch.tensor): Modified Bessel function of the first kind (N, 1).
    See also:
        https://mathworld.wolfram.com/ModifiedBesselFunctionoftheFirstKind.html
    """
    device = X.device
    dtype = X.dtype
    if len(X.shape) == 1:
        X = X[:, None]
        N = X.shape[0]
    else:
        N = 1
    # Compute factorial term
    X = X.repeat(1, Nk)
    K = torch.arange(0, Nk, dtype=dtype, device=device)
    K = K.repeat(N, 1)
    K_factorial = (K + 1).lgamma().exp()
    if order == 0:
        # ..0th order
        i = torch.sum((0.25 * X ** 2) ** K / (K_factorial ** 2), dim=1, dtype=torch.float64)
    else:
        # ..1st order
        i = torch.sum(
            0.5 * X * ((0.25 * X ** 2) ** K /
                       (K_factorial * torch.exp(torch.lgamma(K + 2)))), dim=1, dtype=torch.float64)
    return i
