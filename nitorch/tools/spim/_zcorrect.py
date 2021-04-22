"""
Fit the depth decay model f(z) = s(z) / (1 + b * z**2) + eps
"""
import torch
from nitorch import spatial
from nitorch.core import utils
import math


def zcorrect_square(x, decay=None, sigma=None, lam=10,
             max_iter=128, tol=1e-6, verbose=False):
    """Correct the z signal decay in a SPIM image.

    The signal is modelled as: f(z) = s(z) / (1 + b * z**2) + eps
    where z=0 is the top slice, s(z) is the theoretical signal if there
    was no absorption and b is the decay coefficient.

    Parameters
    ----------
    x : (..., nz) tensor
        SPIM image with the z dimension last and the z=0 plane first
    decay : float, optional
        Initial guess for decay parameter. Default: educated guess.
    sigma : float, optional
        Noise standard deviation. Default: educated guess.
    lam : float, default=10
        Regularisation.
    max_iter : int, default=128
    tol : float, default=1e-6
    verbose : int or bool, default=False

    Returns
    -------
    y : tensor
        Corrected image
    decay : float
        Decay parameters

    """

    x = torch.as_tensor(x)
    if not x.dtype.is_floating_point:
        x = x.to(dtype=torch.get_default_dtype())
    backend = utils.backend(x)
    shape = x.shape
    nz = shape[-1]
    x = x.reshape([-1, nz])
    b = decay

    # decay educated guess: closed form two values at z=1/3 and z=2/3
    z1 = nz//3
    z2 = 2*nz//3
    x1 = x[:, z1].median()
    x2 = x[:, z2].median()
    z1 = float(z1)**2
    z2 = float(z2)**2
    b = b or (x2 - x1) / (x1 * z1 - x2 * z2) 
    b = abs(b)
    y0 = x1 * (1 + b * z1)

    y0 = y0.item()
    b = b.item() if torch.is_tensor(b) else b

    # noise educated guess: assume SNR=5 at z=1/2
    sigma = sigma or (y0 / (1 + b * (nz/2)**2))/5
    lam = lam**2 * sigma**2
    reg = lambda y: spatial.regulariser(y[:, None], membrane=lam, dim=1)[:,0]
    solve = lambda h, g: spatial.solve_field_sym(h[:, None], g[:, None], membrane=lam, dim=1)[:,0]

    print(y0, b, sigma, lam)
    
    # init
    z2 = torch.arange(nz, **backend).square_()
    logy = torch.full_like(x, y0).log_()
    logb = torch.as_tensor(b, **backend)
    y = logy.exp()
    b = logb.exp()
    ll0 = (y / (1 + b * z2) - x).square_().sum() + (logy * reg(logy)).sum()
    ll1 = ll0
    for it in range(max_iter):

        # exponentiate
        y = torch.exp(logy, out=y)
        fit = y / (1 + b * z2)
        res = fit - x

        # compute objective
        ll = res.square().sum() + (logy * reg(logy)).sum()
        gain = (ll1 - ll) / ll0
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{it:3d} | {ll:12.6g} | gain = {gain:12.6g}', end=end)
        if it > 0 and gain < tol:
            break
        ll1 = ll

        # update decay
        g = -(z2 * b * y) / (z2 * b + 1).square()
        h = b * y - z2 * b.square() * y
        h *= z2 / (z2 * b + 1).pow(3)
        h = h.abs_() * res.abs()
        h += g.square()
        g *= res

        g = g.sum()
        h = h.sum()
        logb -= g / h

        # update fit
        b = torch.exp(logb, out=b)
        fit = y / (1 + b * z2)
        res = fit - x

        # ll = (fit - x).square().sum() + 1e3 * (logy[1:] - logy[:-1]).sum().square()
        # gain = (ll1 - ll) / ll0
        # print(f'{it} | {ll.item()} | {gain.item()}', end='\n')

        # update y
        g = h = y / (z2 * b + 1)
        h = h.abs() * res.abs()
        h += g.square()
        g *= res
        g += reg(logy)
        logy -= solve(h, g)

    y = torch.exp(logy, out=y)
    y = y.reshape(shape)
    x = x * (1 + b * z2)
    x = x.reshape(shape)
    return y, b, x


def zcorrect_exp(x, decay=None, sigma=None, lam=10,
                 max_iter=128, tol=1e-6, verbose=False):
    """Correct the z signal decay in a SPIM image.

    The signal is modelled as: f(z) = s(z) exp(-b * z) + eps
    where z=0 is the top slice, s(z) is the theoretical signal if there
    was no absorption and b is the decay coefficient.

    Parameters
    ----------
    x : (..., nz) tensor
        SPIM image with the z dimension last and the z=0 plane first
    decay : float, optional
        Initial guess for decay parameter. Default: educated guess.
    sigma : float, optional
        Noise standard deviation. Default: educated guess.
    lam : float, default=10
        Regularisation.
    max_iter : int, default=128
    tol : float, default=1e-6
    verbose : int or bool, default=False

    Returns
    -------
    y : tensor
        Corrected image
    decay : float
        Decay parameters

    """

    x = torch.as_tensor(x)
    if not x.dtype.is_floating_point:
        x = x.to(dtype=torch.get_default_dtype())
    backend = utils.backend(x)
    shape = x.shape
    nz = shape[-1]
    x = x.reshape([-1, nz])
    b = decay

    # decay educated guess: closed form two values at z=1/3 and z=2/3
    z1 = nz // 3
    z2 = 2 * nz // 3
    x1 = x[:, z1].median()
    x2 = x[:, z2].median()
    z1 = float(z1) ** 2
    z2 = float(z2) ** 2
    b = b or (x2.log() - x1.log()) / (z1 - z2)
    b = abs(b)
    y0 = x1 * torch.as_tensor(b*z1).exp().item()

    y0 = y0.item()
    b = b.item() if torch.is_tensor(b) else b

    # noise educated guess: assume SNR=5 at z=1/2
    sigma = sigma or (y0 * math.exp(-b * (nz / 2) ** 2)) / 5
    lam = lam ** 2 * sigma ** 2
    reg = lambda y: spatial.regulariser(y[:, None], membrane=lam, dim=1)[:, 0]
    solve = lambda h, g: spatial.solve_field_sym(h[:, None], g[:, None], membrane=lam, dim=1)[:, 0]

    print(y0, b, sigma, lam)

    # init
    z = torch.arange(nz, **backend)
    logy = torch.full_like(x, y0).log_()
    logb = torch.as_tensor(b, **backend)
    y = logy.exp()
    b = logb.exp()
    ll0 = (y * (-b * z).exp_() - x).square_().sum() + (logy * reg(logy)).sum()
    ll1 = ll0
    for it in range(max_iter):

        # exponentiate
        y = torch.exp(logy, out=y)
        fit = y * (-b * z).exp_()
        res = fit - x

        # compute objective
        ll = res.square().sum() + (logy * reg(logy)).sum()
        gain = (ll1 - ll) / ll0
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{it:3d} | {ll:12.6g} | gain = {gain:12.6g}', end=end)
        if it > 0 and gain < tol:
            break
        ll1 = ll

        # update decay
        g = -z*b*fit
        h = z*b*fit*(z*b - 1)
        h = h.abs_() * res.abs()
        h += g.square()
        g *= res

        g = g.sum()
        h = h.sum()
        logb -= g / h

        # update fit
        b = torch.exp(logb, out=b)
        fit = y * (-b * z).exp_()
        res = fit - x

        # ll = (fit - x).square().sum() + 1e3 * (logy[1:] - logy[:-1]).sum().square()
        # gain = (ll1 - ll) / ll0
        # print(f'{it} | {ll.item()} | {gain.item()}', end='\n')

        # update y
        g = h = fit
        h = h.abs() * res.abs()
        h += g.square()
        g *= res
        g += reg(logy)
        logy -= solve(h, g)

    y = torch.exp(logy, out=y)
    y = y.reshape(shape)
    x = x * (1 + b * z2)
    x = x.reshape(shape)
    return y, b, x
