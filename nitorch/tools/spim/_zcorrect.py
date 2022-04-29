"""
Fit the depth decay model f(z) = s(z) / (1 + b * z**2) + eps
"""
import torch
from nitorch import spatial
from nitorch.core import utils, py
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
    x = x * (b * z).exp_()
    x = x.reshape(shape)
    return y, b, x


def zcorrect_exp_const(x, decay=None, sigma=None, lam=10, mask=None,
                       max_iter=128, tol=1e-6, verbose=False, snr=5):
    """Correct the z signal decay in a SPIM image.

    The signal is modelled as: f(z) = s * exp(-b * z) + eps
    where z=0 is (arbitrarily) the middle slice, s is the intercept
    and b is the decay coefficient.

    Parameters
    ----------
    x : (..., nz) tensor
        SPIM image with the z dimension last and the z=0 plane first
    decay : float, optional
        Initial guess for decay parameter. Default: educated guess.
    sigma : float, optional
        Noise standard deviation. Default: educated guess.
    lam : float or (float, float), default=10
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
    dim = x.dim() - 1
    nz = shape[-1]
    b = decay

    x = utils.movedim(x, -1, 0).clone()
    if mask is None:
        mask = torch.isfinite(x) & (x > 0)
    else:
        mask = mask & (torch.isfinite(x) & (x > 0))
    x[~mask] = 0

    # decay educated guess: closed form from two values
    if b is None:
        z1 = 2 * nz // 5
        z2 = 3 * nz // 5
        x1 = x[z1]
        x1 = x1[x1 > 0].median()
        x2 = x[z2]
        x2 = x2[x2 > 0].median()
        z1 = float(z1)
        z2 = float(z2)
        b = (x2.log() - x1.log()) / (z1 - z2)
    y = x[(nz-1)//2]
    y = y[y > 0].median().log()

    b = b.item() if torch.is_tensor(b) else b
    y = y.item()
    print(f'init: y = {y}, b = {b}') 

    # noise educated guess: assume SNR=5 at z=1/2
    sigma = sigma or (y / snr)
    lam_y, lam_b = py.make_list(lam, 2)
    lam_y = lam_y ** 2 * sigma ** 2
    lam_b = lam_b ** 2 * sigma ** 2
    reg = lambda t: spatial.regulariser(t, membrane=1, dim=dim, factor=(lam_y, lam_b))
    solve = lambda h, g: spatial.solve_field_fmg(h, g, membrane=1, dim=dim, factor=(lam_y, lam_b))

    # init
    z = torch.arange(nz, **backend) - (nz - 1)/2
    z = utils.unsqueeze(z, -1, dim)
    theta = z.new_empty([2, *x.shape[1:]], **backend)
    logy = theta[0].fill_(y)
    b = theta[1].fill_(b)
    y = logy.exp()
    ll0 = (mask * y * (-b * z).exp_() - x).square_().sum() + (theta * reg(theta)).sum()
    ll1 = ll0

    g = torch.zeros_like(theta)
    h = theta.new_zeros([3, *theta.shape[1:]])
    for it in range(max_iter):

        # exponentiate
        y = torch.exp(logy, out=y)
        fit = (b * z).neg_().exp_().mul_(y).mul_(mask)
        res = fit - x

        # compute objective
        reg_theta = reg(theta)
        ll = res.square().sum() + (theta * reg_theta).sum()
        gain = (ll1 - ll) / ll0
        if verbose:
            end = '\n' if verbose > 1 else '\r'
            print(f'{it:3d} | {ll:12.6g} | gain = {gain:12.6g}', end=end)
        if it > 0 and gain < tol:
            break
        ll1 = ll

        g[0] = (fit * res).sum(0)
        g[1] = -(fit * res * z).sum(0)
        h[0] = (fit * (fit + res.abs())).sum(0)
        h[1] = (fit * (fit + res.abs()) * (z * z)).sum(0)
        h[2] = -(z * fit * fit).sum(0)

        g += reg_theta
        theta -= solve(h, g)

    y = torch.exp(logy, out=y)
    x = x * (b * z).exp_()
    x = utils.movedim(x, 0, -1)
    x = x.reshape(shape)
    return y, b, x


def correct_smooth(x, sigma=None, lam=10, gamma=10, downsample=None,
                   max_iter=16, max_rls=8, tol=1e-6, verbose=False, device=None):
    """Correct the intensity non-uniformity in a SPIM image.

    The signal is modelled as: f = exp(s + b) + eps, with a penalty on
    the (Squared) gradients of s and on the (squared) curvature of b.

    Parameters
    ----------
    x : tensor
        SPIM image with the z dimension last and the z=0 plane first
    sigma : float, optional
        Noise standard deviation. Default: educated guess.
    lam : float, default=10
        Regularisation on the signal.
    gamma : float, default=10
        Regularisation on the bias field.
    max_iter : int, default=16
        Maximum number of Newton iterations.
    max_rls : int, default=8
        Maximum number of reweighting iterations.
        If 1, this is effectively an l2 regularisation.
    tol : float, default=1e-6
        Tolerance for early stopping.
    verbose : int or bool, default=False
        Verbosity level
    device : torch.device, default=x.device
        Use this device during fitting.

    Returns
    -------
    y : tensor
        Fitted image
    bias : float
        Fitted bias
    x : float
        Corrected image

    """

    x = torch.as_tensor(x)
    if not x.dtype.is_floating_point:
        x = x.to(dtype=torch.get_default_dtype())
    dim = x.dim()
    
    # downsampling
    if downsample:
        x0 = x
        downsample = py.make_list(downsample, dim)
        x = spatial.pool(dim, x, downsample)
    shape = x.shape
    x = x.to(device)
    
    # noise educated guess: assume SNR=5 at z=1/2
    center = tuple(slice(s//3, 2*s//3) for s in shape)
    sigma = sigma or x[center].median() / 5
    lam = lam ** 2 * sigma ** 2
    gamma = gamma ** 2 * sigma ** 2
    regy = lambda y, w: spatial.regulariser(y[None], membrane=lam, dim=dim, 
                                            weights=w)[0]
    regb = lambda b: spatial.regulariser(b[None], bending=gamma, dim=dim)[0]
    solvey = lambda h, g, w: spatial.solve_field_sym(h[None], g[None], membrane=lam, dim=dim, 
                                                     weights=w)[0]
    solveb = lambda h, g: spatial.solve_field_sym(h[None], g[None], bending=gamma, dim=dim)[0]

    # init
    l1 = max_rls > 1
    if l1:
        w = torch.ones_like(x)[None]
        llw = w.sum()
        max_rls = 10
    else:
        w = None
        llw = 0
        max_rls = 1
    logb = torch.zeros_like(x)
    logy = x.clamp_min(1e-3).log_()
    y = logy.exp()
    b = logb.exp()
    fit = y * b
    res = fit - x
    llx = res.square().sum()
    lly = (regy(logy, w).mul_(logy)).sum()
    llb = (regb(logb).mul_(logb)).sum()
    ll0 = llx + lly + llb + llw
    ll1 = ll0
    
    for it_ls in range(max_rls):
        for it in range(max_iter):

            # update bias
            g = h = fit
            h = (h*res).abs_()
            h.addcmul_(g, g)
            g *= res
            g += regb(logb)
            logb -= solveb(h, g)
            logb0 = logb.mean()
            logb -= logb0
            logy += logb0

            # update fit / ll
            llb = (regb(logb).mul_(logb)).sum()
            b = torch.exp(logb, out=b)
            y = torch.exp(logy, out=y)
            fit = y * b
            res = fit - x

            # update y
            g = h = fit
            h = (h*res).abs_()
            h.addcmul_(g, g)
            g *= res
            g += regy(logy, w)
            logy -= solvey(h, g, w)
            
            # update fit / ll
            y = torch.exp(logy, out=y)
            fit = y * b
            res = fit - x
            lly = (regy(logy, w).mul_(logy)).sum()
            
            # compute objective
            llx = res.square().sum()
            ll = llx + lly + llb + llw
            gain = (ll1 - ll) / ll0
            ll1 = ll
            if verbose:
                end = '\n' if verbose > 1 else '\r'
                pre = f'{it_ls:3d} | ' if l1 else ''
                print(pre + f'{it:3d} | {ll:12.6g} | gain = {gain:12.6g}', end=end)
            if it > 0 and abs(gain) < tol:
                break

        if l1:
            w, llw = spatial.membrane_weights(logy[None], lam, dim=dim, return_sum=True)
            ll0 = ll
    if verbose:
        print('')
    
    if downsample:
        b = spatial.resize(logb.to(x0.device)[None, None], downsample, shape=x0.shape, anchor='f')[0, 0].exp_()
        y = spatial.resize(logy.to(x0.device)[None, None], downsample, shape=x0.shape, anchor='f')[0, 0].exp_()
        x = x0
    else:
        y = torch.exp(logy, out=y)
    x = x / b
    return y, b, x
