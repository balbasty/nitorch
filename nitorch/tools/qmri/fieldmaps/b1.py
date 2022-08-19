import torch
from torch import Tensor
from typing import Optional
from nitorch.core import py, utils
from nitorch import spatial
import math as pymath
from ...img_statistics import estimate_noise
from ..relax.utils import nll_chi, nll_gauss
from typing import List


def flash_b1_dict(x, fa, tr, r1=(0, 5), b1=(0, 2), n=128):
    """

    Parameters
    ----------
    x : (C, *shape) tensor, Multi flip angle FLASH
    fa : list[float], Flip angle (in deg) of each scan
    tr : float or list[float], TR (in sec) of each scam
    r1 : pair[float] or tensor, bounds of the dictionary (or full range)
    b1 : pair[float] or tensor, bounds of the dictionary (or full range)
    n : int, Number of values in the dictionary, per dimension

    Returns
    -------
    s : (*spatial) tensor
        T2*-weighted PD map
    r1 : (*spatial) tensor
        R1 map
    b1 : (*spatial) tensor
        B1+ map

    """

    backend = dict(dtype=x.dtype, device=x.device)
    tr = py.make_list(tr, len(x))
    fa = utils.make_vector(fa, len(x), dtype=torch.double)
    fa = fa.mul_(pymath.pi / 180).tolist()
    if isinstance(r1, (list, tuple)):
        r1 = torch.linspace(r1[0], r1[1], n)
    if isinstance(b1, (list, tuple)):
        b1 = torch.linspace(b1[0], b1[1], n)
    r1 = torch.as_tensor(r1).tolist()
    b1 = torch.as_tensor(b1).tolist()
    fa = torch.as_tensor(fa, **backend)
    tr = torch.as_tensor(tr, **backend)

    pd, r1, b1 = _flash_dict_r1b1(x, fa, tr, r1, b1)

    return pd, r1, b1


@torch.jit.script
def _flash_dict_r1b1(x, fa, tr, r1_range: List[float], b1_range: List[float]):

    r1 = torch.ones(x.shape[1:], dtype=x.dtype, device=x.device)
    b1 = torch.ones(x.shape[1:], dtype=x.dtype, device=x.device)
    l = torch.full(x.shape[1:], float('inf'), dtype=x.dtype, device=x.device)

    for _ in range(x.dim() - 1):
        fa = fa.unsqueeze(-1)
        tr = tr.unsqueeze(-1)

    for n_iter in range(20):
        print(n_iter+1)

        fa1 = b1 * fa
        s = fa1.sin()
        c = fa1.cos()
        for r1_value in r1_range:
            e1 = (-r1_value * tr).exp()
            y = s * (1 - e1) / (1 - c * e1)
            y = y.abs()
            new_l = -0.5 * (y*x).sum(0).square() / (y*y).sum(0)
            mask = new_l < l
            r1 = r1.masked_fill(mask, r1_value)
            l = torch.where(mask, new_l, l)

        e1 = (-r1 * tr).exp()
        for b1_value in b1_range:
            fa1 = b1_value * fa
            y = fa1.sin() * (1 - e1) / (1 - fa1.cos() * e1)
            y = y.abs()
            new_l = -0.5 * (y*x).sum(0).square() / (y*y).sum(0)
            mask = new_l < l
            b1 = b1.masked_fill(mask, b1_value)
            l = torch.where(mask, new_l, l)

    e1 = (-r1 * tr).exp()
    fa1 = b1 * fa
    y = fa1.sin() * (1 - e1) / (1 - fa1.cos() * e1)
    y = y.abs()
    pd = (y * x).sum(0) / (y * y).sum(0)

    return pd, r1, b1


def flash_b1(x, fa, tr, lam=(0, 0, 100), penalty=('m', 'm', 'b'),
             chi=False, pd=None, r1=None, b1=None):
    """Estimate B1+ from Variable flip angle data

    Parameters
    ----------
    x : (C, *spatial) tensor
        Input flash images
    fa : (C,) sequence[float]
        Flip angle (in deg)
    tr : float or (C,) sequence[float]
        Repetition time (in sec)
    lam : (float, float, float), default=(0, 0, 10)
        Regularization value for the T2*w Signal, T1  map and B1 map.
    penalty : 3x {'membrane', 'bending'}, default=('m', 'm', 'b')
        Regularization type for the T2*w Signal, T1  map and B1 map.

    Returns
    -------
    s : (*spatial) tensor
        T2*-weighted PD map
    r1 : (*spatial) tensor
        R1 map
    b1 : (*spatial) tensor
        B1+ map

    """

    tr = py.make_list(tr, len(x))
    fa = utils.make_vector(fa, len(x), dtype=torch.double)
    fa = fa.mul_(pymath.pi / 180).tolist()
    lam = py.make_list(lam, 3)
    penalty = py.make_list(penalty, 3)

    sd, df, mu = 0, 0, []
    for x1 in x:
        bg, fg = estimate_noise(x1, chi=True)
        sd += bg['sd'].log()
        df += bg['dof'].log()
        mu += [fg['mean']]
    sd = (sd / len(x)).exp()
    df = (df / len(x)).exp()
    prec = 1/(sd*sd)
    if not chi:
        df = 1

    i = fa.index(min(fa))
    j = fa.index(min(fa[:i] + fa[i+1:]))
    init_r1 = mu[i] * fa[i] / tr[i] - mu[j] * fa[j] / tr[j]
    init_r1 /= mu[j] / fa[j] - mu[i] / fa[i]
    init_r1 /= 2
    init_pd = mu[i] * mu[j] * (tr[i] * fa[j] / fa[i] - tr[j] * fa[i] / fa[j])
    init_pd /= mu[j] * tr[i] * fa[j] - mu[i] * tr[j] * fa[i]

    shape = x.shape[1:]
    theta = x.new_empty([3, *shape])
    theta[0] = init_pd.log() if pd is None else pd.clamp_min(sd*1e-8).log()
    theta[1] = init_r1.log() if r1 is None else r1.clamp_min(1e-8).log()
    theta[2] = 0 if b1 is None else b1.clamp_min(sd*1e-8).log()
    n = (x != 0).sum()

    g = torch.zeros_like(theta)
    h = theta.new_zeros([6, *theta.shape[1:]])
    g1 = torch.zeros_like(theta)
    h1 = theta.new_zeros([6, *theta.shape[1:]])

    prm = dict(
        membrane=(lam[0] if penalty[0][0] == 'm' else 0,
                  lam[1] if penalty[1][0] == 'm' else 0,
                  lam[2] if penalty[2][0] == 'm' else 0),
        bending=(lam[0] if penalty[0][0] == 'b' else 0,
                 lam[1] if penalty[1][0] == 'b' else 0,
                 lam[2] if penalty[2][0] == 'b' else 0),
    )

    ll0 = lr0 = float('inf')
    iter_start = 3
    for n_iter in range(32):

        if df > 1 and n_iter == iter_start:
            ll0 = lr0 = float('inf')

        # derivatives of likelihood term
        df1 = 1 if n_iter < iter_start else df
        ll, g, h = derivatives(x, fa, tr, theta[0], theta[1], theta[2],
                               prec, df1, g, h, g1, h1)

        # derivatives of regularization term
        reg = spatial.regulariser(theta, **prm, absolute=1e-10)
        g += reg
        lr = 0.5 * dot(theta, reg)

        l, l0 = ll + lr, ll0 + lr0
        gain = l0 - l
        print(f'{n_iter:2d} | {ll/n:12.6g} + {lr/n:12.6g} = {l/n:12.6g} | gain = {gain/n:12.6g}')

        # Gauss-Newton update
        h[:3] += 1e-8 * h[:3].abs().max(0).values
        h[:3] += 1e-5
        # if n_iter % 2:
        #     delta = torch.zeros_like(theta)
        #     prm1 = dict(membrane=prm['membrane'][:2], bending=prm['bending'][:2])
        #     hh = torch.stack([h[0], h[1], h[3]])
        #     delta[:2] = spatial.solve_field(hh, g[:2], **prm1, max_iter=16,
        #                                     absolute=1e-10)
        #     del hh
        # else:
        #     delta = torch.zeros_like(theta)
        #     prm1 = dict(membrane=prm['membrane'][-1], bending=prm['bending'][-1])
        #     delta[-1:] = spatial.solve_field(h[2:3], g[2:3], **prm1, max_iter=16,
        #                                      absolute=1e-10)
        delta = spatial.solve_field(h, g, **prm, max_iter=16, absolute=1e-10)
        # theta -= spatial.solve_field_fmg(h, g, **prm)

        # line search
        dd = spatial.regulariser(delta, **prm, absolute=1e-10)
        dt = dot(dd, theta)
        dd = dot(dd, delta)
        success = False
        armijo = 1
        theta0 = theta
        ll0, lr0 = ll, lr
        for n_ls in range(12):
            theta = theta0.sub(delta, alpha=armijo)
            ll = nll(x, fa, tr, *theta, prec, df1)
            lr = 0.5 * armijo * (armijo * dd - 2 * dt)
            if ll + lr < ll0:  # and theta[1].max() < 0.69:
                print(n_ls, 'success', ((ll + lr)/n).item(), (ll0/n).item())
                success = True
                break
            print(n_ls, 'failure', ((ll + lr)/n).item(), (ll0/n).item())
            armijo /= 2
        if not success and n_iter > 5:
            theta = theta0
            break

        # delta = spatial.solve_field_fmg(h, g, **prm)
        #
        # # line search
        # dd = spatial.regulariser(delta, **prm)
        # dt = dot(dd, theta)
        # dd = dot(dd, delta)
        # success = False
        # armijo = 1
        # theta0 = theta
        # ll0, lr0 = ll, lr
        # for n_ls in range(12):
        #     theta = theta0.sub(delta, alpha=armijo)
        #     ll = nll(x, fa, tr, theta[0], theta[1], theta[2]) * prec
        #     lr = 0.5 * armijo * (armijo * dd - 2 * dt)
        #     if ll + lr < ll0:
        #         print('success', n_ls)
        #         success = True
        #         break
        #     armijo /= 2
        # if not success:
        #     theta = theta0
        #     break

        # import matplotlib.pyplot as plt
        # plt.subplot(2, 2, 1)
        # plt.imshow(theta[0, :, :, theta.shape[-1]//2].exp())
        # plt.axis('off')
        # plt.colorbar()
        # plt.title('PD * exp(-TE * R2*)')
        # plt.subplot(2, 2, 2)
        # plt.imshow(theta[1, :, :, theta.shape[-1]//2].exp())
        # plt.axis('off')
        # plt.colorbar()
        # plt.title('R1')
        # plt.subplot(2, 2, 3)
        # plt.imshow(theta[2, :, :, theta.shape[-1]//2].exp())
        # plt.axis('off')
        # plt.colorbar()
        # plt.title('B1+')
        # plt.show()

        # vmin, vmax = 0, 2 * sum(mu) / len(mu)
        # y = flash_signals(fa, tr, *theta)
        # ex = 3
        # plt.rcParams["figure.figsize"] = (4, len(x)+ex)
        # for i in range(len(x)):
        #     plt.subplot(len(x) + ex, 4, 4 * i + 1)
        #     plt.imshow(x[i, ..., x.shape[-1] // 2].cpu(), vmin=vmin, vmax=vmax)
        #     plt.axis('off')
        #     if i == 0: plt.title('Obs')
        #     plt.subplot(len(x) + ex, 4, 4 * i + 2)
        #     plt.imshow(y[i, ..., x.shape[-1] // 2].cpu(), vmin=vmin, vmax=vmax)
        #     plt.axis('off')
        #     if i == 0: plt.title('Fit')
        #     plt.subplot(len(x) + ex, 4, 4 * i + 3)
        #     plt.imshow((x[i, ..., x.shape[-1] // 2] -
        #                 y[i, ..., y.shape[1] // 2]).cpu(),
        #                cmap=plt.get_cmap('coolwarm'))
        #     plt.axis('off')
        #     if i == 0: plt.title('Diff')
        #     plt.colorbar()
        #     plt.subplot(len(x) + ex, 4, 4 * i + 4)
        #     plt.imshow(((theta[-1, ..., x.shape[-1] // 2].exp() * fa[i]) / pymath.pi).cpu())
        #     plt.axis('off')
        #     if i == 0: plt.title('B1')
        #     plt.colorbar()
        # all_fa = torch.linspace(0, 2*pymath.pi, 512)
        # loc = [(theta.shape[1]//2, theta.shape[2]//2, theta.shape[3]//2),
        #        (2*theta.shape[1]//3, theta.shape[2]//2, theta.shape[3]//2),
        #        (theta.shape[1]//3, theta.shape[2]//3, theta.shape[3]//2)]
        # for j, (nx, ny, nz) in enumerate(loc):
        #     plt.subplot(len(x) + ex,  1, len(x) + j + 1)
        #     plt.plot(all_fa, flash_signals(all_fa, tr[:1]*512, *theta[:, nx, ny, nz]).cpu())
        #     plt.scatter(fa, x[:, nx, ny, nz].cpu())
        # plt.show()

        if gain < 1e-4 * n:
            break

    theta.exp_()
    return theta[0], theta[1], theta[2]


def demo_flash_b1(x, fa, tr, prec=1, df=1):
    tr = py.make_list(tr, len(x))
    fa = utils.make_vector(fa, len(x), dtype=torch.double)
    fa = fa.mul_(pymath.pi / 180).tolist()

    shape = x.shape[1:]
    theta = x.new_empty([3, *shape])
    theta[0] = 5
    theta[1] = 1
    theta[2] = 0
    n = (x != 0).sum()

    g = torch.zeros_like(theta)
    h = theta.new_zeros([6, *theta.shape[1:]])
    g1 = torch.zeros_like(theta)
    h1 = theta.new_zeros([6, *theta.shape[1:]])

    ll0 = float('inf')
    iter_start = 3
    for n_iter in range(32):

        if df > 1 and n_iter == iter_start:
            ll0 = float('inf')

        # derivatives of likelihood term
        df1 = 1 if n_iter < iter_start else df
        ll, g, h = derivatives(x, fa, tr, theta[0], theta[1], theta[2],
                               prec, df1, g, h, g1, h1)

        gain = ll0 - ll
        print(f'{n_iter:2d} | {ll/n:12.6g} | gain = {gain/n:12.6g}')

        # Gauss-Newton update
        h[:3] += 1e-8 * h[:3].abs().max(0).values
        h[:3] += 1e-5
        delta = spatial.solve_field_closedform(h[:, None, None, None],
                                               g[:, None, None, None],
                                               absolute=1e-10)[:, 0, 0, 0]

        # line search
        success = False
        armijo = 1
        theta0 = theta
        ll0 = ll
        for n_ls in range(12):
            theta = theta0.sub(delta, alpha=armijo)
            ll = nll(x, fa, tr, *theta, prec, df1)
            if ll < ll0: # and theta[1].max() < 0.69:
                print(n_ls, 'success', ((ll)/n).item(), (ll0/n).item())
                success = True
                break
            print(n_ls, 'failure', ((ll)/n).item(), (ll0/n).item())
            armijo /= 2
        if not success and n_iter > 5:
            theta = theta0
            break

        import matplotlib.pyplot as plt
        all_fa = torch.linspace(0, 2 * pymath.pi, 512)
        plt.plot(all_fa, flash_signals(all_fa, tr[:1] * 512, *theta))
        plt.scatter(fa, x)
        plt.show()

        if gain < 1e-4 * n:
            break

    theta.exp_()
    return theta[0], theta[1], theta[2]


def derivatives(x, fa, tr, pd, r1, b1, prec, df=1,
                g=None, h=None, g1=None, h1=None):

    # derivatives
    if g is None:
        g = pd.new_zeros([3, *pd.shape])
    else:
        g.zero_()
    if h is None:
        h = pd.new_zeros([6, *pd.shape])
    else:
        h.zero_()
    if g1 is None:
        g1 = pd.new_zeros([3, *pd.shape])
    else:
        g1.zero_()
    if h1 is None:
        h1 = pd.new_zeros([6, *pd.shape])
    else:
        h1.zero_()

    l = 0
    for x1, fa1, tr1 in zip(x, fa, tr):
        y1, g1, h1 = flash_derivatives(fa1, tr1, pd, r1, b1, g1, h1)
        l1, g1, h1 = chain_rule(x1, y1, g1, h1, prec, df)
        l += l1
        g += g1
        h += h1

    return l, g, h


# @torch.jit.script
def flash_derivatives(fa: float, tr: float, pd, r1, b1,
                      g: Optional[Tensor], h: Optional[Tensor]):

    # exponentiate
    pd = pd.exp()
    r1 = r1.exp()
    b1 = b1.exp()

    # reconstruct signal
    e = (-tr * r1).exp()
    b1 = b1 * fa
    s, c = b1.sin(), b1.cos()
    y = pd * s * (1 - e) / (1 - c * e)
    sgn = y.sign()

    # derivatives
    if g is None:
        g = pd.new_zeros([3] + pd.shape)
    if h is None:
        h = pd.new_zeros([6] + pd.shape)

    g[:] = y
    g[1] *= -e / (1 - e) * (c * (1 - e) / (1 - c * e) - 1) * (tr * r1)
    g[2] *= b1 * (c / s - e * s / (1 - c * e))
    h[:3] = g
    h[1] *= 1 - (tr * r1) * (1 + c * e) / (1 - c * e)
    h[2] *= (1 + b1 * (c / s - 2 * e * s / (1 - c * e))
               - b1 / ((c / s - e * s / (1 - c * e)) * (s*s)))

    y *= sgn
    g *= sgn
    # h *= sgn
    h.abs_()

    msk = torch.isfinite(h).bitwise_not_().any(0)
    msk = msk | torch.isfinite(g).bitwise_not_().any(0)
    g[:, msk] = 0
    h[:, msk] = 0
    return y, g, h


def chain_rule(x, y, g, h, prec, df=1):
    nll = nll_gauss if df == 1 else nll_chi
    df = [] if df == 1 else [df]

    msk = x == 0
    y.masked_fill_(msk, 0)
    l, r = nll(x.clone(), y, ~msk, prec, *df, return_residuals=True)

    h[:3] *= r.abs()
    h[:3].addcmul_(g, g)
    torch.mul(g[0], g[1], out=h[3])
    torch.mul(g[0], g[2], out=h[4])
    torch.mul(g[1], g[2], out=h[5])
    g *= r

    g *= prec
    h *= prec
    h.masked_fill_(msk, 0)
    return l, g, h


def nll(x, fa, tr, pd, r1, b1, prec, df=1):
    nll = nll_gauss if df == 1 else nll_chi
    df = [] if df == 1 else [df]

    l = 0
    for x1, fa1, tr1 in zip(x, fa, tr):
        y = flash_signal(fa1, tr1, pd, r1, b1).masked_fill_(x1 == 0, 0)
        l += nll(x1.clone(), y, x1 != 0, prec, *df, return_residuals=False)
    return l


@torch.jit.script
def flash_signal(fa: float, tr: float, pd, r1, b1):
    # exponentiate
    pd = pd.exp()
    r1 = r1.exp()
    b1 = b1.exp()

    # reconstruct signal
    e = (-tr * r1).exp()
    b1 = b1 * fa
    s, c = b1.sin(), b1.cos()
    y = pd * s * (1 - e) / (1 - c * e)
    return y.abs()


def flash_signals(fa, tr, pd, r1, b1):
    y = pd.new_empty([len(fa), *pd.shape])
    for y1, fa1, tr1 in zip(y, fa, tr):
        y1[...] = flash_signal(fa1, tr1, pd, r1, b1)
    return y


# ======================================================================
#                       SINUSOID FIT (LONG TR)
# ======================================================================


def sin_b1(x, fa, lam=(0, 1e3), penalty=('m', 'b'), chi=True, pd=None, b1=None):
    """Estimate B1+ from Variable flip angle data with long TR

    Parameters
    ----------
    x : (C, *spatial) tensor
        Input flash images
    fa : (C,) sequence[float]
        Flip angle (in deg)
    lam : (float, float), default=(0, 1e4)
        Regularization value for the T2*w Signal, T1  map and B1 map.
    penalty : 3x {'membrane', 'bending'}, default=('m', 'b')
        Regularization type for the T2*w Signal, T1  map and B1 map.

    Returns
    -------
    s : (*spatial) tensor
        T2*-weighted PD map
    b1 : (*spatial) tenmsor
        B1+ map

    """

    fa = utils.make_vector(fa, len(x), dtype=torch.double)
    fa = fa.mul_(pymath.pi / 180).tolist()
    lam = py.make_list(lam, 2)
    penalty = py.make_list(penalty, 2)

    sd, df, mu = 0, 0, 0
    for x1 in x:
        bg, fg = estimate_noise(x1, chi=True)
        sd += bg['sd'].log()
        df += bg['dof'].log()
        mu += fg['mean'].log()
    sd = (sd / len(x)).exp()
    df = (df / len(x)).exp()
    mu = (mu / len(x)).exp()
    prec = 1/(sd*sd)
    if not chi:
        df = 1

    # mask low SNR voxels
    # x = x * (x > 5 * sd)

    shape = x.shape[1:]
    theta = x.new_empty([2, *shape])
    theta[0] = mu.log() if pd is None else pd.log()
    theta[1] = 0 if b1 is None else b1.log()
    n = (x != 0).sum()

    g = torch.zeros_like(theta)
    h = theta.new_zeros([3, *theta.shape[1:]])
    g1 = torch.zeros_like(theta)
    h1 = theta.new_zeros([3, *theta.shape[1:]])

    prm = dict(
        membrane=[lam[0] if penalty[0][0] == 'm' else 0,
                  lam[1] if penalty[1][0] == 'm' else 0],
        bending=[lam[0] if penalty[0][0] == 'b' else 0,
                 lam[1] if penalty[1][0] == 'b' else 0],
    )

    lam0 = dict(membrane=prm['membrane'][-1], bending=prm['bending'][-1])
    ll0 = lr0 = factor = float('inf')
    for n_iter in range(32):

        if n_iter == 1:
            ll0 = lr0 = float('inf')

        # decrease regularization
        factor, factor_prev = 1 + 10 ** (5 - n_iter), factor
        factor_ratio = factor / factor_prev if n_iter else float('inf')
        prm['membrane'][-1] = lam0['membrane'] * factor
        prm['bending'][-1] = lam0['bending'] * factor

        # derivatives of likelihood term
        df1 = 1 if n_iter == 0 else df
        ll, g, h = sin_full_derivatives(x, fa, theta[0], theta[1],
                                        prec, df1, g, h, g1, h1)

        # derivatives of regularization term
        reg = spatial.regulariser(theta, **prm)
        g += reg
        lr = 0.5 * dot(theta, reg)

        l, l0 = ll + lr, ll0 + factor_ratio * lr0
        gain = l0 - l
        print(f'{n_iter:2d} | {ll/n:12.6g} + {lr/n:12.6g} = {l/n:12.6g} | gain = {gain/n:12.6g}')

        # Gauss-Newton update
        h[:2] += 1e-8 * h[:2].abs().max(0).values
        h[:2] += 1e-5
        delta = spatial.solve_field_fmg(h, g, **prm)

        mx = delta.abs().max()
        if mx > 64:
            delta *= 64 / mx

        # theta -= delta
        # ll0, lr0 = ll, lr

        # line search
        dd = spatial.regulariser(delta, **prm)
        dt = dot(dd, theta)
        dd = dot(dd, delta)
        success = False
        armijo = 1
        theta0 = theta
        ll0, lr0 = ll, lr
        for n_ls in range(12):
            theta = theta0.sub(delta, alpha=armijo)
            ll = sin_nll(x, fa, theta[0], theta[1], prec, df1)
            lr = 0.5 * armijo * (armijo * dd - 2 * dt)
            if ll + lr < ll0: # and theta[1].max() < 0.69:
                print(n_ls, 'success', ((ll + lr)/n).item(), (ll0/n).item())
                success = True
                break
            print(n_ls, 'failure', ((ll + lr)/n).item(), (ll0/n).item())
            armijo /= 2
        if not success and n_iter > 5:
            theta = theta0
            break

        import matplotlib.pyplot as plt

        # plt.subplot(1, 2, 1)
        # plt.imshow(theta[0, :, :, theta.shape[-1]//2].exp())
        # plt.colorbar()
        # plt.subplot(1, 2, 2)
        # plt.imshow(theta[1, :, :, theta.shape[-1]//2].exp())
        # plt.colorbar()
        # plt.show()

        plt.rcParams["figure.figsize"] = (4, len(x))
        vmin, vmax = 0, 2 * mu
        y = sin_signals(fa, *theta)
        ex = 3
        for i in range(len(x)):
            plt.subplot(len(x) + ex, 4, 4 * i + 1)
            plt.imshow(x[i, ..., x.shape[-1] // 2], vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.subplot(len(x) + ex, 4, 4 * i + 2)
            plt.imshow(y[i, ..., x.shape[-1] // 2], vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.subplot(len(x) + ex, 4, 4 * i + 3)
            plt.imshow(x[i, ..., x.shape[-1] // 2] -
                       y[i, ..., y.shape[1] // 2],
                       cmap=plt.get_cmap('coolwarm'))
            plt.axis('off')
            plt.colorbar()
            plt.subplot(len(x) + ex, 4, 4 * i + 4)
            plt.imshow((theta[-1, ..., x.shape[-1] // 2].exp() * fa[i]) / pymath.pi)
            plt.axis('off')
            plt.colorbar()
        all_fa = torch.linspace(0, 2*pymath.pi, 512)
        loc = [(theta.shape[1]//2, theta.shape[2]//2, theta.shape[3]//2),
               (2*theta.shape[1]//3, theta.shape[2]//2, theta.shape[3]//2),
               (theta.shape[1]//3, theta.shape[2]//3, theta.shape[3]//2)]
        for j, (nx, ny, nz) in enumerate(loc):
            plt.subplot(len(x) + ex,  1, len(x) + j + 1)
            plt.plot(all_fa, sin_signals(all_fa, *theta[:, nx, ny, nz]))
            plt.scatter(fa, x[:, nx, ny, nz])
        plt.show()

        # if gain < 1e-4 * n:
        #     break

    theta.exp_()
    return theta[0], theta[1]


def sin_full_derivatives(x, fa, pd, b1, prec, df=1,
                         g=None, h=None, g1=None, h1=None):

    # derivatives
    if g is None:
        g = pd.new_zeros([2, *pd.shape])
    if h is None:
        h = pd.new_zeros([3, *pd.shape])
    if g1 is None:
        g1 = pd.new_zeros([2, *pd.shape])
    if h1 is None:
        h1 = pd.new_zeros([3, *pd.shape])

    l = 0
    for x1, fa1 in zip(x, fa):
        y1, g1, h1 = sin_derivatives(fa1, pd, b1, g1, h1)
        l1, g1, h1 = sin_chain_rule(x1, y1, g1, h1, prec, df)
        l += l1
        g += g1
        h += h1

    return l, g, h


@torch.jit.script
def sin_derivatives(fa: float, pd, b1,
                    g: Optional[Tensor], h: Optional[Tensor]):
    # exponentiate
    pd = pd.exp()
    b1 = b1.exp()

    # reconstruct signal
    b1 = b1 * fa
    s, c = b1.sin(), b1.cos()
    y = pd * s
    sgn = y.sign()

    # derivatives
    if g is None:
        g = pd.new_zeros([2] + pd.shape)
    if h is None:
        h = pd.new_zeros([3] + pd.shape)

    g[:] = y
    g[1] *= b1 * c / s
    h[:2] = g
    h[2] *= 1 - b1 * s / c

    y *= sgn
    g *= sgn
    # h *= sgn
    h.abs_()
    return y, g, h


def sin_chain_rule(x, y, g, h, prec, df=1):
    nll = nll_gauss if df == 1 else nll_chi
    df = [] if df == 1 else [df]

    msk = x == 0
    y.masked_fill_(msk, 0)
    l, r = nll(x.clone(), y, ~msk, prec, *df, return_residuals=True)

    h[:2] *= r.abs()
    h[:2] += g.square()
    h[2] = g[0] * g[1]
    g *= r

    h.masked_fill_(msk, 0)

    g *= prec
    h *= prec
    return l, g, h


def sin_nll(x, fa, pd, b1, prec, df=1):
    nll = nll_gauss if df == 1 else nll_chi
    df = [] if df == 1 else [df]
    l = 0
    for x1, fa1 in zip(x, fa):
        y = sin_signal(fa1, pd, b1)
        y.masked_fill_(x1 == 0, 0)
        l += nll(x1.clone(), y, x1 != 0, prec, *df, return_residuals=False)
    return l


@torch.jit.script
def sin_signal(fa: float, pd, b1):
    # exponentiate
    pd = pd.exp()
    b1 = b1.exp()

    # reconstruct signal
    b1 = b1 * fa
    y = pd * b1.sin()
    return y.abs_()


def sin_signals(fa, pd, b1):
    y = pd.new_empty([len(fa), *pd.shape])
    for y1, fa1 in zip(y, fa):
        y1[...] = sin_signal(fa1, pd, b1)
    return y


# ======================================================================
#                                HELPERS
# ======================================================================


@torch.jit.script
def dot(x, y):
    return x.flatten().dot(y.flatten())
