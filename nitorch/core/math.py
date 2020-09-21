# -*- coding: utf-8 -*-
"""Mathematical functions not available in standard torch.

Created on Thu Jun 18 14:04:00 2020

@author: mikael.brudfors@gmail.com
"""


from numpy import real
from scipy.linalg import expm as expm_scipy
from scipy.linalg import logm as logm_scipy
import torch


def round(t, decimals=0):
    """ Round a tensor to the given number of decimals.

    Args:
        t (torch.tensor) Tensor.
        decimals (int, optional): Round to this decimal, defaults to zero.

    Returns:
        t (torch.tensor): Rounded tensor.

    """
    return torch.round(t * 10 ** decimals) / (10 ** decimals)


def expm(M):
    """ Computes the matrix exponential of M.

    Args:
        M (torch.tensor): Square matrix (N, N)

    Returns:
        M (torch.tensor): Matrix exponential (N, N)

    """
    device = M.device
    dtype = M.dtype
    M = M.cpu().numpy()
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
    M = M.cpu().numpy()
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
