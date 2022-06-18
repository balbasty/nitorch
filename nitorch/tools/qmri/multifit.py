import torch
from nitorch import spatial
import math

"""

spgr : list[dict{
            dat: (N, *shape) tensor, Multi-echo Spoiled Gradient Echo
            fa: float, Nominal flip angle (deg)
            tr: float, Repetition time (s)
            te: list[float], Echo times (s)
            bandwidth: float, Readout bandwidth (hz/pixel)
            readout: int, Readout direction
            polarity: {1, -1}, Polarity of first echo
            sigma: float, Noise standard deviation
    }]
se : list[dict{
            dat: (Ne, *shape) tensor, Multi-echo Spin Echo
            te: list[float], Echo times (s)
            bandwidth: float, Readout bandwidth (hx/pixel)
            readout: int, Readout direction
            polarity: {1, -1}, Polarity of first echo
            sigma: float, Noise standard deviation
    }]
fmaps : dict{
            b0: dict{
                magnitude: (*shape) tensor, Magnitude of the B0 fieldmap
                phase: (*shape) tensor, Phase of the B0 fieldmap
                sigma: float, Noise standard deviation
                delta_te: float, TE difference (s)
                affine : (D+1, D+1) tensor, Orientation matrix
            }
            b1+: (*shape) tensor, B1+ transmit fieldmap
            b1-: (*shape) tensor, B1- receive fieldmap
        }
theta : dict{
            dat: (K, *shape) tensor,
                rho: (*shape) tensor, Log proton density
                r1: (*shape) tensor, Log longitudinal relaxation rate (s)
                r2: (*shape) tensor, Log transverse relaxation rate (s)
                b0: (*shape) tensor, Delta B0 [local] (Hz)
                b0smo: (*shape) tensor,  Delta B0 [smooth/global] (Hz)
            indices : ['rho', 'r1', 'r2', 'b0', 'b0smooth']
            affine : (D+1, D+1) tensor, Orientation matrix
        }
wrls : (K|1, *shape) tensor, (J)TV weight map
lam : dict{rho, r1, r2, b0, b0smo}, Regularization factor
prm : dict{
            gamma: float, gyromagnetic ratio,
            max_inner: int, Max number of inner Gauss-Newton iterations
            max_outer: int, Max number of outer Gauss-Newton iterations
            tol: float, Tolerance for early stopping
        }

"""

def run_multifit(spgr, se, fmaps, theta, wrls, lam, prm):

    for _ in range(max(1, prm['max_inner'])):
        l, tv = run_multifit_inner(spgr, se, fmaps, theta, wrls, lam, prm)

    for _ in range(max(1, prm['max_outer'])):
        l, b, tv = run_multifit_outer(spgr, se, fmaps, theta, wrls, lam, prm)

    return l, b, tv


def run_multifit_outer(spgr, se, fmaps, theta, wrls, lam, prm):

    shape = spgr[0]['dat'].shape[1:]
    backend = dict(dtype=spgr[0]['dat'].dtype, device=spgr[0]['dat'].device)

    l = 0
    g = torch.empty([1, *shape], **backend)
    h = torch.empty([1, *shape], **backend)
    indices = theta['indices'][-1]

    # --- likelihood ---------------------------------------------------

    for spgr1 in spgr:
        l1, g1, h1 = derivatives_dist_spgr(spgr1, fmaps, theta, prm)
        l += l1
        g += g1
        h += h1

    for se1 in se:
        l1, g1, h1 = derivatives_dist_se(se1, fmaps, theta, prm)
        l += l1
        g += g1
        h += h1

    l1, g1, h1 = derivatives_fmap(fmaps['b0'], theta, prm)
    l += l1
    g += g1
    h += h1

    # --- prior --------------------------------------------------------
    rg = spatial.regulariser(theta['dat'][-1:], weights=wrls,
                             bending=lam['b0smooth'])
    r = 0.5 * dot(theta['dat'][-1:], rg)
    g += rg

    # --- solve GN step ------------------------------------------------
    delta = spatial.solve_field(h, g, bending=lam['b0smooth'])

    dd = spatial.regulariser(delta, weights=wrls, bending=lam['b0smooth'])
    rd = dot(dd, theta['dat'][-1:])
    dd = dot(dd, delta)

    # --- line search --------------------------------------------------
    armijo, armijo_prev, success = 1, 0, False
    theta0, wrls0 = theta['dat'].clone(), wrls.clone()
    for n_ls in range(min(1, prm['max_ls'])):
        theta['dat'][:-1].sub_(delta, alpha=(armijo - armijo_prev))
        armijo_prev = armijo
        new_r = 0.5 * armijo * (armijo * dd - 2 * rd)
        for _ in range(min(1, prm['max_inner'])):
            new_l, new_tv = run_multifit_inner(spgr, se, fmaps, theta, wrls, lam, prm)
        if new_l + new_r < l:
            success = True
            break
        else:
            armijo /= 2

    if not success:
        theta['dat'].copy(theta0)
        return l, r, None

    l, r, tv = new_l, r + new_r, new_tv
    if prm['verbose']:
        total = l + r + tv
        print(f'{l.item():12.6g} + {r.item():12.6g} + {tv.item():12.6g} '
              f'= {total.item():12.6g}')

    return l, r, tv


def run_multifit_inner(spgr, se, fmaps, theta, wrls, lam, prm):

    shape = spgr[0]['dat'].shape[1:]
    backend = dict(dtype=spgr[0]['dat'].dtype, device=spgr[0]['dat'].device)

    l = 0
    g = torch.empty([4, *shape], **backend)
    h = torch.empty([10, *shape], **backend)
    indices = theta['indices'][:4]

    # --- likelihood ---------------------------------------------------

    for spgr1 in spgr:
        l1, g1, h1 = derivatives_spgr(spgr1, fmaps, theta, prm)
        l += l1
        g += g1
        h += h1

    for se1 in se:
        l1, g1, h1 = derivatives_se(se1, fmaps, theta, prm)
        l += l1
        g[:3] += g1
        h[:3] += h1[:3]             # diagonal
        h[[4, 5, 7]] += h1[3:]      # off diagonal

    l1, g1, h1 = derivatives_fmap(fmaps['b0'], theta, prm)
    l += l1
    g[3] += g1
    h[3] += h1

    # --- prior --------------------------------------------------------
    g += spatial.regulariser(theta['dat'][:4], weights=wrls,
                             membrane=1, factor=[lam[k] for k in indices])

    # --- update theta -------------------------------------------------
    theta['dat'][:4] -= spatial.solve_field(h, g, weights=wrls, membrane=1,
                                            factor=[lam[k] for k in indices])

    # --- update TV ----------------------------------------------------
    tv = update_tv(theta, wrls, lam, prm)

    return l, tv


def derivatives_spgr(spgr, fmaps, theta, prm):

    rho, r1, r2, b0, b0smo = theta['dat']
    rho, r1, r2, b0 = rho.exp(), r1.exp(), r2.exp(), b0 + b0smo
    e1 = r1.mul(-spgr['tr']).exp_()

    # compute signal at TE=0
    s0 = rho
    if 'b1-' in fmaps:
        s0 *= fmaps['b1-'].dat
    if 'b1+' in fmaps:
        fa = fmaps['b1-'].dat * spgr['fa']
        c, s = fa.cos(), fa.sin_()
        ce1 = c.mul_(e1)
    else:
        fa = spgr['fa']
        c, s = math.cos(fa), math.sin(fa)
        ce1 = c * e1
    s0 *= s
    s0 *= (1 - e1)
    s0 /= (1 - ce1)
    s0.abs_()

    # gradient and hessian of signal
    g = s.new_zeros([4, *s.shape[1:]])
    h = s.new_zeros([10, *s.shape[1:]])

    phi_pos, phi_neg = make_distortion(b0, spgr)
    residuals, l = 0, 0
    for e, te in enumerate(spgr['te']):

        # compute R2* decay
        e2 = b0.abs().add_(r2).mul_(-te).exp_()

        # distort and residuals
        s = s0 * e2
        phi = phi_neg if (e % 2) else phi_pos
        s = dist_pull(s, phi, spgr)
        r = s - spgr['dat'][e]
        l += 0.5 * ssq(r)
        r = dist_push(s, phi, iphi, spgr)
        residuals += r

        # R2/B0 gradient
        g[2] += -te * e2 * r  # term common to R2 and B0

    @torch.jit.script
    def g_r1(e1, ce1, r1, tr: float):
        return tr * r1 * (e1 - ce1) / ((1 - e1) * (1 - ce1))

    # gradient
    g[0] = 1
    g[1] = g_r1(e1, ce1, r1, spgr['tr'])
    g[3] = g[2]
    g[2] *= r2
    g[3] *= prm['gamma'] * b0.sign()
    g[:2] *= residuals
    g *= s0




