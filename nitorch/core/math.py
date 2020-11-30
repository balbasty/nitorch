"""Mathematical functions."""

# mikael.brudfors@gmail.com
# yael.balbastre@gmail.com

import torch
from .constants import inf, ninf


def round(t, decimals=0):
    """ Round a tensor to the given number of decimals.

    Args:
        t (torch.tensor) Tensor.
        decimals (int, optional): Round to this decimal, defaults to zero.

    Returns:
        t (torch.tensor): Rounded tensor.

    """
    return torch.round(t * 10 ** decimals) / (10 ** decimals)


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
    if input.requires_grad:
        zero = torch.as_tensor(0, dtype=input.dtype, device=input.device)
        input = torch.where(mask, zero, input)
    else:
        input[mask] = 0
    return torch.sum(input, *args, **kwargs)


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
    if input.requires_grad:
        zero = torch.as_tensor(0, dtype=input.dtype, device=input.device)
        input = torch.where(mask, zero, input)
    else:
        input[mask] = 0
    mask = ~mask
    weights = mask.sum(*args, **kwargs).to(kwargs.get('dtype', input.dtype))
    return torch.sum(input, *args, **kwargs) / weights


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
    if not inplace:
        input = input.clone()
    mask = torch.isnan(input)
    if requires_grad:
        zero = torch.as_tensor(0, dtype=input.dtype, device=input.device)
        input = torch.where(mask, zero, input)
    else:
        input[mask] = 0
    mask = ~mask
    weights = mask.sum(*args, **kwargs).to(kwargs.get('dtype', input.dtype))
    mean = torch.sum(input, *args, **kwargs) / weights
    input = input.square() if requires_grad else input.square_()
    var = torch.sum(input, *args, **kwargs) / weights
    if requires_grad:
        var = var - mean
        if unbiased:
            var = var * weights / (weights - 1)
    else:
        var -= mean
        if unbiased:
            weights /= (weights - 1)
            var *= weights
    return var


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


def nanmin(input, *args, inplace=False, **kwargs):
    """Compute the minimum of a tensor, excluding nans.

    Notes
    -----
    .. This function cannot compute the minimum of two tensors, it only
       computes the minimum of one tensor (along a dimension).
    .. If all values (across a dimension) are nans, the output value
       will be inf.


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
    out : tensor, optional
        Output placeholder.

    Returns
    -------
    values : tensor
        Output tensor
    indices : tensor[long], if `dim is not None`
        Index location of each minimum value found

    """
    # TODO: minimum of two tensors
    input = torch.as_tensor(input)
    mask = torch.isnan(input)
    if inplace and not input.requires_grad:
        input[mask] = inf
    else:
        val_inf = torch.as_tensor(inf, dtype=input.dtype, device=input.device)
        input = torch.where(mask, val_inf, input)
    return torch.min(input, *args, **kwargs)


def nanmax(input, *args, inplace=False, **kwargs):
    """Compute the maximum of a tensor, excluding nans.

    Notes
    -----
    .. This function cannot compute the maximum of two tensors, it only
       computes the maximum of one tensor (along a dimension).
    .. If all values (across a dimension) are nans, the output value
       will be -inf.

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
    out : tensor, optional
        Output placeholder.

    Returns
    -------
    values : tensor
        Output tensor
    indices : tensor[long], if `dim is not None`
        Index location of each maximum value found

    """
    # TODO: minimum of two tensors
    input = torch.as_tensor(input)
    mask = torch.isnan(input)
    if inplace and not input.requires_grad:
        input[mask] = ninf
    else:
        val_ninf = torch.as_tensor(ninf, dtype=input.dtype, device=input.device)
        input = torch.where(mask, val_ninf, input)
    return torch.max(input, *args, **kwargs)


def softmax(Z, dim=-1, get_ll=False, W=None):
    """ SoftMax (safe).

    Parameters
    ----------
    Z : torch.tensor
        Tensor with values.
    dim : int, default=-1
        Dimension to take softmax, defaults to last dimensions.
    get_ll : bool, default=False
        Compute log-likelihood, defaults to False.
    W : torch.tensor, optional:
        Observation weights.

    Returns
    -------
    Z : torch.tensor
        Soft-maxed tensor with values.

    """
    Z_max, _ = torch.max(Z, dim=dim)
    Z = torch.exp(Z - Z_max[:, None])
    Z_sum = torch.sum(Z, dim=dim)
    if get_ll:
        # Compute log-likelihood
        if W is None:
            ll = torch.sum(torch.log(Z_sum) + Z_max, dtype=torch.float64)
        else:
            ll = torch.sum((torch.log(Z_sum) + Z_max)*W.squeeze(), dtype=torch.float64)
    else:
        ll = None
    Z = Z / Z_sum[:, None]
    return Z, ll


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


def besseli(X, order=0, Nk=50):
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
