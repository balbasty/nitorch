import torch
from nitorch import spatial, io
from nitorch.core import utils, py, linalg
from nitorch.tools.img_statistics import estimate_noise
import math

"""

spgr : list[dict{
            dat: (N, *shape) tensor, Multi-echo Spoiled Gradient Echo
            fa: float, Nominal flip angle (deg)
            tr: float, Repetition time (s)
            te: list[float], Echo times (s)
            bandwidth: float, Readout bandwidth (Hz/pixel)
            readout: int, Readout direction
            polarity: {+1, -1}, Polarity of first echo
            bipolar: bool, Bipolar readout (echoes have alternate polarity)
            sigma: float, Noise standard deviation
    }]
se : list[dict{
            dat: (Ne, *shape) tensor, Multi-echo Spin Echo
            te: list[float], Echo times (s)
            bandwidth: float, Readout bandwidth (hx/pixel)
            readout: int, Readout direction
            polarity: {1, -1}, Polarity of first echo
            bipolar: bool, Bipolar readout (echoes have alternate polarity)
            sigma: float, Noise standard deviation
    }]
fmaps : dict{
            b0: dict{
                magnitude: (*shape) tensor, Magnitude of the B0 fieldmap
                phase: (*shape) tensor, Phase of the B0 fieldmap
                sigma: float, Noise standard deviation
                delta_te: float, TE difference (s)
                bandwidth: float, Readout bandwidth (hx/pixel)
                readout: int, Readout direction
                polarity: {1, -1}, Polarity of first echo
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
            max_inner: int, Max number of inner Gauss-Newton iterations [5]
            max_outer: int, Max number of outer Gauss-Newton iterations [10]
            tol: float, Tolerance for early stopping, [1e-5]
            joint: bool, Joint TV, [True]
            field: float, Field strength, [3]
            device: torch.device
        }

"""


def multifit(spgr=None, se=None, bssfp=None, fmaps=None, lam=None, prm=None, **kwargs):
    prm = prm or {}
    prm.update(kwargs)
    prm.setdefault('max_inner', 5)
    prm.setdefault('max_outer', 10)
    prm.setdefault('max_ls', 12)
    prm.setdefault('tol', 1e-5)
    prm.setdefault('joint', True)
    prm.setdefault('field', 3)
    prm.setdefault('device', 'cpu')
    prm.setdefault('verbose', True)
    device = dict(device=prm.get('device', None))

    lam = lam or {}
    lam.setdefault('rho', 5)
    lam.setdefault('r1', 5)
    lam.setdefault('r2', 5)
    lam.setdefault('r2p', 0.1)
    lam.setdefault('scl', 5)
    lam.setdefault('b0', 10)

    lam['b0'] /= (50 * prm['field'])

    fmaps = fmaps or {}
    spgr = list(py.make_sequence(spgr or []))
    se = list(py.make_sequence(se or []))
    bssfp = list(py.make_sequence(bssfp or []))
    if len(spgr) == 0 or (len(se) == 0 and len(bssfp) == 0):
        raise ValueError('Multifit requires SPGR and either SE or bSSFP acquisitions')

    for spgr1 in spgr:
        spgr1['dat'] = map_files(spgr1['dat'])
        spgr1['sigma'] = 1
        spgr1['mu'] = []
        for e, dat in enumerate(spgr1['dat']):
            noise, tissue = estimate_noise(load(dat, **device))
            spgr1['sigma'] *= noise['sd'].item()
            spgr1['mu'] += [tissue['mean'].item()]
        spgr1['sigma'] = spgr1['sigma'] ** (1/len(spgr1['dat']))
    for bssfp1 in bssfp:
        bssfp1['dat'] = map_files(bssfp1['dat'])
        bssfp1['sigma'] = 1
        bssfp1['mu'] = []
        for e, dat in enumerate(bssfp1['dat']):
            noise, tissue = estimate_noise(load(dat, **device))
            bssfp1['sigma'] *= noise['sd'].item()
            bssfp1['mu'] += [tissue['mean'].item()]
        bssfp1['sigma'] = bssfp1['sigma'] ** (1/len(bssfp1['dat']))
    for se1 in se:
        se1['dat'] = map_files(se1['dat'])
        se1['sigma'] = 1
        se1['mu'] = []
        for e, dat in enumerate(se1['dat']):
            noise, tissue = estimate_noise(load(dat, **device))
            se1['sigma'] *= noise['sd'].item()
            se1['mu'] += [tissue['mean'].item()]
        se1['sigma'] = se1['sigma'] ** (1/len(se1['dat']))
    if 'b1+' in fmaps:
        fmaps['b1+'] = map_files(fmaps['b1+'], nobatch=True)
    if 'b1-' in fmaps:
        fmaps['b1-'] = map_files(fmaps['b1-'], nobatch=True)
    if 'b0' in fmaps:
        fmaps['b0']['phase'] = map_files(fmaps['b0']['phase'], nobatch=True)
        fmaps['b0']['magnitude'] = map_files(fmaps['b0']['magnitude'], nobatch=True)

    pd, r1, r2s = init_spgr(spgr)
    if se:
        scl, r2 = init_se(se, pd, r1)
    else:
        scl, r2 = [], init_bssfp(bssfp, pd, r1)

    shape = spgr[0]['dat'].shape[1:]
    if hasattr(spgr[0]['dat'], 'affine'):
        affine = spgr[0]['dat'][0].affine
    else:
        affine = spatial.affine_default(shape)
    theta0 = torch.zeros([5 + len(se), *shape], device=prm['device'])
    theta0[0] = math.log(pd)
    theta0[1] = math.log(r1)
    theta0[2] = math.log(r2)
    theta0[3] = math.log(r2s - r2)
    for i, scl1 in enumerate(scl):
        theta0[5+i] = math.log(scl1)
    indices = ['rho', 'r1', 'r2', 'r2p', 'b0']
    indices += ['scl'] * len(se)
    theta = dict(dat=theta0, affine=affine, indices=indices)
    wrls = torch.ones([1 if prm['joint'] else 4, *shape], device=prm['device'])

    l, b, tv = run_multifit(spgr, se, bssfp, fmaps, theta, wrls, lam, prm)
    return theta


def init_spgr(spgr):

    # log-linear fit per contrast
    s0, r2s = [], []
    for spgr1 in spgr:
        mu = torch.as_tensor(spgr1['mu']).log()
        te = -torch.as_tensor(spgr1['te'])
        s01, r2s1 = torch.stack([torch.ones(len(mu)), te], -1).pinverse() @ mu
        s0.append(s01.exp().item())
        r2s.append(r2s1.item())
    r2s = sum(r2s) / len(r2s)

    # small flip angle approximation
    s0 = s0[:2]
    fa = [spgr1['fa'] for spgr1 in spgr[:2]]
    tr = [spgr1['tr'] for spgr1 in spgr[:2]]
    r1 = s0[0] * fa[0] / tr[0] - s0[1] * fa[1] / tr[1]
    r1 /= 2 * (s0[1] / fa[1] - s0[0] / fa[0])
    pd = s0[0] * s0[1] * (tr[1] * fa[0] / fa[1] - tr[0] * fa[1] / fa[0])
    pd /= s0[0] * tr[1] * fa[0] - s0[1] * tr[0] * fa[1]

    return pd, r1, r2s


def init_se(se, pd, r1):

    # log-linear fit per contrast
    scl, r2 = [], []
    for se1 in se:
        mu = torch.as_tensor(se1['mu']).log()
        te = -torch.as_tensor(se1['te'])
        s01, r21 = torch.stack([torch.ones(len(mu)), te], -1).pinverse() @ mu
        s01_fit = pd * (1 - math.exp(-se1['tr'] * r1))
        scl.append(s01.exp().item() / s01_fit)
        r2.append(r21.item())
    r2 = sum(r2) / len(r2)

    return scl, r2


def init_bssfp(acq, pd, r1):
    r2 = []
    for acq1 in acq:
        mu = acq1['mu'][0]
        te = acq1.get('te', [acq1['tr']/2])[0]
        fa = acq1['fa']
        pd = pd * math.sin(fa) / mu
        r21 = r1 * (pd - math.cos(fa) - 1) / (pd * r1 * te - math.cos(fa) + 1)
        r2.append(r21)
    r2 = sum(r2) / len(r2)

    return r2


def map_files(files, nobatch=False, keep_open=False):
    """Concat and map input files or tensors"""
    if isinstance(files, str):
        files = io.volumes.map(files, keep_open=keep_open)
        if files.dim == 5 and files.shape[3] == 1 and files.dim == 5:
            # channel along the 5th dimension (C), drop 4th (T) dimension
            files = files[:, :, :, 0, :]
        elif files.dim == 5 and files.shape[4] == 1:
            # channel along the 4th dimension (T), drop 5th (C) dimension
            files = files[:, :, :, :, 0]
        elif files.dim == 4:
            # channel along the 4th dimension (T)
            pass
        elif files.dim == 3:
            files = files[..., None]
        else:
            raise ValueError(f'Unsupported input shape {list(files.shape)}')
        files = files.movedim(-1, 0)
        if nobatch:
            files = files[0]
    elif isinstance(files, (list, tuple)):
        files = list(map(map_files, files))
        if isinstance(files[0], io.MappedArray):
            files = io.cat(files)
        else:
            files = torch.cat(files)
        if nobatch:
            files = files[0]
    return files


def load(dat, rand=True, missing=0, **backend):
    """Load a tensor from a file (if input is a mapped file)"""
    if torch.is_tensor(dat):
        return dat.to(**backend)
    elif isinstance(dat, io.MappedArray):
        dat = dat.fdata(rand=rand, missing=missing, **backend)
        dat.masked_fill_(torch.isfinite(dat).logical_not_(), 0)
        return dat
    else:
        return dat


def run_multifit(spgr, se, bssfp, fmaps, theta, wrls, lam, prm):

    l = b = tv = None
    n = theta['dat'].numel() // len(theta['dat'])

    if 'b0' in fmaps:
        for _ in range(3):
            run_b0fit(fmaps, theta, lam, prm)

    for _ in range(10): # range(max(1, prm['max_inner'])):
        l, tv = run_multifit_inner(spgr, se, bssfp, fmaps, theta, wrls, lam, prm)
        if prm['verbose']:
            total = l + tv
            print(f'{l.item()/n:12.6g} + {tv.item()/n:12.6g} '
                  f'= {total.item()/n:12.6g}')

    # for _ in range(max(1, prm['max_outer'])):
    #     l, b, tv = run_multifit_outer(spgr, se, fmaps, theta, wrls, lam, prm)

    return l, b, tv


def run_b0fit(fmaps, theta, lam, prm):

    if 'b0' not in fmaps:
        return None, None
    i = theta['indices'].index('b0')

    # --- likelihood ---------------------------------------------------
    l, g, h = derivatives_fmap(fmaps['b0'], theta, prm)
    g, h = g[None], h[None]
    n = g.numel() // len(g)

    # --- prior --------------------------------------------------------
    rg = spatial.regulariser(theta['dat'][None, i], membrane=lam['b0'], absolute=1e-6)
    r = 0.5 * dot(theta['dat'][None, i], rg)
    g += rg

    # --- solve GN step ------------------------------------------------
    delta = spatial.solve_field_fmg(h, g, membrane=lam['b0'], absolute=1e-6)

    dd = spatial.regulariser(delta, membrane=lam['b0'], absolute=1e-6)
    rd = dot(dd, theta['dat'][None, i])
    dd = dot(dd, delta)

    # --- line search --------------------------------------------------
    armijo, armijo_prev, success = 1, 0, False
    theta0 = theta['dat'].clone()
    for n_ls in range(max(1, prm['max_ls'])):
        theta['dat'][i].sub_(delta[0], alpha=(armijo - armijo_prev))
        armijo_prev = armijo
        new_r = 0.5 * armijo * (armijo * dd - 2 * rd)
        for _ in range(max(1, prm['max_inner'])):
            new_l = loss_fmap(fmaps['b0'], theta, prm)
        if new_l + new_r < l:
            success = True
            break
        else:
            armijo /= 2

    if not success:
        theta['dat'].copy_(theta0)
        return l, r

    l, r = new_l, r + new_r
    if prm['verbose']:
        total = l + r
        print(f'{l.item()/n:12.6g} + {r.item()/n:12.6g} = {total.item()/n:12.6g}')

    return l, r


def run_multifit_outer(spgr, se, fmaps, theta, wrls, lam, prm):

    shape = spgr[0]['dat'].shape[1:]
    backend = dict(dtype=spgr[0]['dat'].dtype, device=prm.get('device', None))

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

    if 'b0' in fmaps:
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
    for n_ls in range(max(1, prm['max_ls'])):
        theta['dat'][-1:].sub_(delta, alpha=(armijo - armijo_prev))
        armijo_prev = armijo
        new_r = 0.5 * armijo * (armijo * dd - 2 * rd)
        for _ in range(max(1, prm['max_inner'])):
            new_l, new_tv = run_multifit_inner(spgr, se, fmaps, theta, wrls, lam, prm)
        if new_l + new_r < l:
            success = True
            break
        else:
            theta['dat'][:-1] = theta0[:-1]
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


def run_multifit_inner(spgr, se, bssfp, fmaps, theta, wrls, lam, prm):

    shape = spgr[0]['dat'].shape[1:]
    backend = dict(dtype=torch.float32,
                   device=prm.get('device', None) or
                          getattr(spgr[0]['dat'], 'device', None))

    l = 0
    k = 5 + len(se)
    k2 = (k*(k+1))//2
    g = torch.empty([k, *shape], **backend)
    h = torch.empty([k2, *shape], **backend)

    # --- likelihood ---------------------------------------------------
    o = len(se)
    indices = torch.eye(5 + len(se), dtype=torch.long)
    indices.diagonal(0, -1, -2).copy_(torch.arange(5 + len(se)))
    c = 5 + len(se)
    for i in range(5 + len(se)):
        for j in range(i+1, 5 + len(se)):
            indices[i, j] = indices[j, i] = c
            c += 1

    ind_g_spgr = slice(4)
    ind_h_spgr = list(sorted(set(indices[:4, :4].flatten().tolist())))
    for spgr1 in spgr:
        l1, g1, h1 = derivatives_spgr(spgr1, fmaps, theta, prm)
        l += l1
        g[ind_g_spgr] += g1
        h[ind_h_spgr] += h1

    for i, se1 in enumerate(se):
        ind_g_se = [0, 1, 2, 5+i]
        ind_h_se = list(sorted(set(indices[ind_g_se, :][:, ind_g_se].flatten().tolist())))
        l1, g1, h1 = derivatives_se(i, se1, fmaps, theta, prm)
        l += l1
        g[ind_g_se] += g1
        h[ind_h_se] += h1

    ind_g_bssfp = slice(3)
    ind_h_bssfp = list(sorted(set(indices[:3, :3].flatten().tolist())))
    for i, bssfp1 in enumerate(bssfp):
        l1, g1, h1 = derivatives_bssfp(bssfp1, fmaps, theta, prm)
        l += l1
        g[ind_g_bssfp] += g1
        h[ind_h_bssfp] += h1

    if 'b0' in fmaps:
        l1, g1, h1 = derivatives_fmap(fmaps['b0'], theta, prm)
        l += l1
        g[4] += g1
        h[4] += h1

    # --- prior --------------------------------------------------------
    g += spatial.regulariser(theta['dat'], weights=wrls, membrane=1,
                             factor=[lam[k] for k in theta['indices']])

    # --- update theta -------------------------------------------------
    h[:5+o] += 1e-3 * h[:5+o].abs().max(0).values
    msk = (linalg.sym_det(h.movedim(0, -1)) <= 0)
    if msk.any():
        h[:5+o, msk] += 1e-3 * h[:5+o, msk].abs().max(0).values
        msk = (linalg.sym_det(h.movedim(0, -1)) <= 0)
        if msk.any():
            h[:5+o, msk] = 1e-3 * h[:5+o, msk].abs().max(0).values
            h[5+o:, msk] = 0
            msk = (linalg.sym_det(h.movedim(0, -1)) <= 0)
            if msk.any():
                h[:5+o, msk] = 1e-5
                h[5+o:, msk] = 0
    msk = ~(torch.isfinite(g).all(0) & torch.isfinite(h).all(0))
    if msk.any():
        g[:, msk] = 0
        h[:5+o, msk] = 1e-5
        h[5+o:, msk] = 0

    delta = spatial.solve_field(h, g, weights=wrls, membrane=1,
                                factor=[lam[k] for k in theta['indices']])
    delta[~delta.isfinite()] = 0
    theta['dat'] -= delta

    # --- update TV ----------------------------------------------------
    tv = update_tv(theta, wrls, lam, prm)

    return l, tv


def derivatives_spgr(spgr, fmaps, theta, prm):
    """Derivatives for a Spoiled Gradient Echo signal (SPGR)"""
    device = dict(device=prm.get('device', None))

    rho, r1, r2, r2p, b0 = theta['dat'][:5]
    rho, r1, r2, r2p = rho.exp(), r1.exp(), r2.exp(), r2p.exp()
    masknans_(rho)
    masknans_(r1)
    masknans_(r2)
    masknans_(r2p)
    e1 = r1.mul(-spgr['tr']).exp_()
    masknans_(e1)

    # compute signal at TE=0
    s0 = rho
    if 'b1-' in fmaps:
        s0 *= load(fmaps['b1-'], **device)
    if 'b1+' in fmaps:
        fa = load(fmaps['b1+'] * spgr['fa'], **device)
        c, s = fa.cos(), fa.sin_()
        ce1 = c.mul_(e1)
    else:
        fa = spgr['fa']
        c, s = math.cos(fa), math.sin(fa)
        ce1 = c * e1
    s0 *= s

    @torch.jit.script
    def make_s0(s0, e1, ce1):
        return s0.mul_((1 - e1)/(1 - ce1))

    s0 = make_s0(s0, e1, ce1).abs_()
    masknans_(s0)

    # gradient and hessian of signal
    g = s0.new_zeros([4, *s0.shape])
    h = s0.new_zeros([10, *s0.shape])

    # precompute grad/hess component wrt log(R1)
    @torch.jit.script
    def g_r1(e1, ce1, r1, tr: float):
        return tr * r1 * (e1 - ce1) / ((1 - e1) * (1 - ce1))
    @torch.jit.script
    def h_r1(ce1, r1, tr: float):
        return 1 - tr * r1 * (1 + ce1) / (1 - ce1)
    g1 = g_r1(e1, ce1, r1, spgr['tr'])
    h1 = h_r1(ce1, r1, spgr['tr']).abs_().mul_(g1)
    masknans_(g1)
    masknans_(h1)

    phi_pos, phi_neg, jac_pos, jac_neg = make_distortion(b0, spgr, prm)
    l = 0
    for e, te in enumerate(spgr['te']):

        # compute R2* decay
        e2 = r2.add(r2p).mul_(-te).exp_()
        masknans_(e2)

        # distort and residuals
        r = s0 * e2                             # undistorted fit
        phi = phi_neg if (e % 2) else phi_pos   # distortion
        jac = jac_neg if (e % 2) else jac_pos   # pile-up
        r = dist_pull(r, phi, jac, spgr)        # distorted fit
        d = load(spgr['dat'][e], **device)      # obs
        m = torch.isfinite(d).bitwise_not_()    # mask
        m = m.bitwise_or_(d <= 0)               # mask
        r = r.sub_(d).masked_fill_(m, 0)        # residuals
        l += 0.5 * ssq(r)                       # log-likelihood
        del d

        m = m.bitwise_not_().to(s0)
        m = dist_push(m, phi, jac, spgr).mul_(s0)
        a = dist_push(r.abs(), phi, jac, spgr).mul_(e2)
        r = dist_push(r, phi, jac, spgr).mul_(e2)

        # hessian of signal x absolute residuals
        h[0].add_(a)                                            # PD**2
        h[1].addcmul_(a, h1)                                    # R1**2
        h[2].addcmul_(a, (te * r2) * (te * r2 - 1).abs_())      # R2**2
        h[3].addcmul_(a, (te * r2p) * (te * r2p - 1).abs_())    # R2'**2
        del a

        # squared gradient of signal ("Gauss-Newton")
        e2 = e2.square_().mul_(m)
        h[0].add_(e2)                                           # PD*PD
        h[1].addcmul_(e2, g1.square())                          # R1*R1
        h[2].addcmul_(e2, r2.square(), value=te*te)             # R2*R2
        h[3].addcmul_(e2, r2p.square(), value=te*te)            # R2'*R2'
        h[4].addcmul_(e2, g1)                                   # PD*R1
        h[5].addcmul_(e2, r2, value=-te)                        # PD*R2
        h[6].addcmul_(e2, r2p, value=-te)                       # PD*R2'
        h[7].addcmul_(e2, g1*r2, value=-te)                     # R1*R2
        h[8].addcmul_(e2, g1*r2p, value=-te)                    # R1*R2'
        h[9].addcmul_(e2, r2*r2p, value=te*te)                  # R2*R2'
        del e2

        # gradient
        g[0].add_(r)                                            # PD
        g[1].addcmul_(r, g1)                                    # R1
        g[2].addcmul_(r, r2, value=-te)                         # R2
        g[3].addcmul_(r, r2p, value=-te)                        # R2'
        del r

    g *= s0
    h *= s0

    # modulate by noise precision
    sigma = spgr['sigma']
    isigma2 = 1 / (sigma*sigma)
    l *= isigma2
    g *= isigma2
    h *= isigma2

    return l, g, h


def derivatives_bssfp(acq, fmaps, theta, prm):
    """Derivatives for a balanced steady-state free precession (bSSFP)"""
    # Assumes TE = TR/2
    device = dict(device=prm.get('device', None))

    tr = acq['tr']
    rho, r1, r2, _, b0 = theta['dat'][:5]
    rho, r1, r2 = rho.exp(), r1.exp(), r2.exp()
    masknans_(rho)
    masknans_(r1)
    masknans_(r2)
    e1 = r1.mul(-tr).exp_()
    e2 = r2.mul(-tr).exp_()
    masknans_(e1)
    masknans_(e2)

    # compute signal at TE=0
    s0 = rho
    if 'b1-' in fmaps:
        s0 *= load(fmaps['b1-'], **device)
    if 'b1+' in fmaps:
        fa = load(fmaps['b1+'] * acq['fa'], **device)
        c, s = fa.cos(), fa.sin_()
    else:
        fa = acq['fa']
        c, s = math.cos(fa), math.sin(fa)
    s0 *= s
    del s

    den = 1 - c * (e1 - e2) - e1 * e2
    s0 *= (1 - e1) * e2.sqrt() / den
    s0 = s0.abs_()
    masknans_(s0)

    # gradient and hessian of signal
    g = s0.new_zeros([3, *s0.shape])
    h = s0.new_zeros([6, *s0.shape])

    phi, _, jac, _ = make_distortion(b0, acq, prm)

    # distort and residuals
    r = dist_pull(s0, phi, jac, acq)        # distorted fit
    d = load(acq['dat'][0], **device)       # obs
    m = torch.isfinite(d).bitwise_not_()    # mask
    m = m.bitwise_or_(d <= 0)               # mask
    r = r.sub_(d).masked_fill_(m, 0)        # residuals
    l = 0.5 * ssq(r)                        # log-likelihood
    del d

    m = m.bitwise_not_().to(s0)
    s0 *= dist_push(m, phi, jac, acq)
    a = dist_push(r.abs(), phi, jac, acq)
    r = dist_push(r, phi, jac, acq)

    # gradient of signal / signal
    g[0].fill_(1)                                                       # PD
    g[1].copy_(tr * r1 * e1 * (1 / (1 - e1) - (c + e2) / den))    # R1
    g[2].copy_(tr * r2 * e2 * (c - e1) / den)                           # R2

    # hessian of signal x absolute residuals
    h[1].copy_((1 - tr * r1) * g[1] - (tr * r1 * e1).square() *
               (1 / (1 - e1).square() - ((c + e2)/den).square()))       # R1**2
    h[2].copy_(- tr * r2 * g[2] + g[2].square())                        # R2**2
    g[2].add_(-0.5 * tr * r2)
    h[2].add_(g[2])
    h[0].fill_(1)
    h[1:3].abs_().addcmul_(g[1:3], g[1:3])
    h[:3] *= a
    del a

    # squared gradient of signal ("Gauss-Newton")
    h[0].add_(1)                                                        # PD*PD
    h[1].addcmul_(g[1], g[1])                                           # R1*R1
    h[2].addcmul_(g[2], g[2])                                           # R2*R2
    h[3].add_(g[1])                                                     # PD*R1
    h[4].add_(g[2])                                                     # PD*R2
    h[5].addcmul_(g[1], g[2])                                           # R1*R2

    # gradient
    g *= r
    del r

    g *= s0
    h *= s0

    # modulate by noise precision
    sigma = acq['sigma']
    isigma2 = 1 / (sigma*sigma)
    l *= isigma2
    g *= isigma2
    h *= isigma2

    return l, g, h


def derivatives_se(i, se, fmaps, theta, prm):
    """Derivatives for a Spin Echo signal (SE)"""
    device = dict(device=prm.get('device', None))
    rho, r1, r2, _, b0 = theta['dat'][:5]
    scl = theta['dat'][5+i]
    rho, r1, r2, scl = rho.exp(), r1.exp(), r2.exp(), scl.exp()
    masknans_(rho)
    masknans_(r1)
    masknans_(r2)
    e1 = r1.mul(-se['tr']).exp_()
    masknans_(e1)

    # compute signal at TE=0
    s0 = rho
    s0 *= scl
    # s0 *= se['scl']
    if 'b1-' in fmaps:
        s0 *= load(fmaps['b1-'], **device)

    @torch.jit.script
    def make_s0(s0, e1):
        return s0.mul_(1 - e1)

    s0 = make_s0(s0, e1).abs_()

    # gradient and hessian of signal
    g = s0.new_zeros([4, *s0.shape])
    h = s0.new_zeros([10, *s0.shape])

    # precompute grad/hess component wrt log(R1)
    @torch.jit.script
    def g_r1(e1, r1, tr: float):
        return tr * r1 * e1 / (1 - e1)
    @torch.jit.script
    def h_r1(r1, tr: float):
        return 1 - tr * r1
    g1 = g_r1(e1, r1, se['tr'])
    h1 = h_r1(r1, se['tr']).abs_().mul_(g1)
    masknans_(g1)
    masknans_(h1)

    phi_pos, phi_neg, jac_pos, jac_neg = make_distortion(b0, se, prm)
    l = 0
    for e, te in enumerate(se['te']):

        # compute R2* decay
        e2 = r2.mul(-te).exp_()
        masknans_(e2)

        # distort and residuals
        r = s0 * e2                             # undistorted fit
        phi = phi_neg if (e % 2) else phi_pos   # distortion
        jac = jac_neg if (e % 2) else jac_pos   # pile-up
        r = dist_pull(r, phi, jac, se)          # distorted fit
        d = load(se['dat'][e], **device)        # obs
        m = torch.isfinite(d).bitwise_not_()    # mask
        m = m.bitwise_or_(d <= 0)               # mask
        r = r.sub_(d).masked_fill_(m, 0)        # residuals
        l += 0.5 * ssq(r)                       # log-likelihood
        del d

        m = m.bitwise_not_().to(s0)
        m = dist_push(m, phi, jac, se).mul_(s0)
        a = dist_push(r.abs(), phi, jac, se).mul_(e2)
        r = dist_push(r, phi, jac, se).mul_(e2)

        # hessian of signal x absolute residuals
        h[0].add_(a)                                            # PD**2
        h[1].addcmul_(a, h1)                                    # R1**2
        h[2].addcmul_(a, (te * r2) * (te * r2 - 1).abs_())      # R2**2
        h[3].add_(a)                                            # SC**2
        del a

        # squared gradient of signal ("Gauss-Newton")
        e2 = e2.square_().mul_(m)
        h[0].add_(e2)                                           # PD*PD
        h[1].addcmul_(e2, g1.square())                          # R1*R1
        h[2].addcmul_(e2, r2.square(), value=te*te)             # R2*R2
        h[3].add_(e2)                                           # SC*SC
        h[4].addcmul_(e2, g1)                                   # PD*R1
        h[5].addcmul_(e2, r2, value=-te)                        # PD*R2
        h[6].add_(e2)                                           # PD*SC
        h[7].addcmul_(e2, g1*r2, value=-te)                     # R1*R2
        h[8].addcmul_(e2, g1)                                   # R1*SC
        h[9].addcmul_(e2, r2, value=-te)                        # R2*SC
        del e2

        # gradient
        g[0].add_(r)                                            # PD
        g[1].addcmul_(r, g1)                                    # R1
        g[2].addcmul_(r, r2, value=-te)                         # R2
        g[3].add_(r)                                            # SC
        del r

    g *= s0
    h *= s0

    # modulate by noise precision
    sigma = se['sigma']
    isigma2 = 1 / (sigma*sigma)
    l *= isigma2
    g *= isigma2
    h *= isigma2

    return l, g, h


def loss_fmap(fmap, theta, prm):
    """Log-likelihood for a fieldmap (delta phase)"""
    device = dict(device=prm.get('device', None))
    b0 = theta['dat'][4]
    shape = fmap['magnitude'].shape

    # forward model
    bw0 = fmap.get('bandwidth', 1)
    mat = theta['affine'].inverse() @ fmap['affine']
    scl = spatial.voxel_size(mat)
    bw = bw0 * scl[fmap.get('readout', -1)]
    fmap['bandwidth'] = bw
    dphi, _, _, _ = make_distortion(b0, fmap, prm)
    b0 = dist_pull(b0, dphi, None, fmap)
    scl *= fmap.get('psf', 1)
    # scl = scl.sub_(1).clamp_min_(0)
    b0 = spatial.smooth(b0, fwhm=scl, dim=b0.dim())
    phi = spatial.affine_grid(mat.to(b0), shape)
    b0 = spatial.grid_pull(b0, phi, bound='dct2', extrapolate=True)
    b0 = b0 * (2*math.pi*fmap['delta_te'])  # to radian

    # loss
    sigma = fmap['sigma']
    m, p = load(fmap['magnitude'], **device), load(fmap['phase'], **device)
    msk = torch.isfinite(p).bitwise_not_()
    msk = msk.logical_or_(torch.isfinite(m).bitwise_not_())
    msk = msk.logical_or_(m <= 0)
    h = m.square().masked_fill_(msk, 0).div_(sigma*sigma)
    r = (b0 - p).cos_().neg_().add_(1).masked_fill_(msk, 0)
    l = dot(h, r)
    return l


def derivatives_fmap(fmap, theta, prm):
    """Derivatives for a Fieldmap"""
    device = dict(device=prm.get('device', None))
    b0 = theta['dat'][4]
    shape0, shape = b0.shape, fmap['magnitude'].shape

    # forward model
    bw0 = fmap.get('bandwidth', 1)
    mat = theta['affine'].inverse() @ fmap['affine']
    scl = spatial.voxel_size(mat)
    bw = bw0 * scl[fmap.get('readout', -1)]
    fmap['bandwidth'] = bw
    dphi, _, _, _ = make_distortion(b0, fmap, prm)
    b0 = dist_pull(b0, dphi, None, fmap)
    scl *= fmap.get('psf', 1)
    # scl = scl.sub_(1).clamp_min_(0)
    b0 = spatial.smooth(b0, fwhm=scl, dim=b0.dim())
    phi = spatial.affine_grid(mat.to(b0), shape)
    b0 = spatial.grid_pull(b0, phi, bound='dct2', extrapolate=True)
    b0 = b0 * (2*math.pi*fmap['delta_te'])

    # gradient and hessian of signal
    sigma = fmap['sigma']
    m, p = load(fmap['magnitude'], **device), load(fmap['phase'], **device)
    msk = torch.isfinite(p).bitwise_not_()
    msk = msk.logical_or_(torch.isfinite(m).bitwise_not_())
    msk = msk.logical_or_(m <= 0)
    h = m.square().masked_fill_(msk, 0).div_(sigma*sigma)
    g = b0 - p
    l = dot(h, g.cos().neg_().add_(1).masked_fill_(msk, 0))
    g = g.sin_().mul_(h)
    del m, p, msk

    # adjoint
    g = spatial.grid_push(g, phi, shape=shape0, bound='dct2', extrapolate=True)
    h = spatial.grid_push(h, phi, shape=shape0, bound='dct2', extrapolate=True)
    g = spatial.smooth(g, fwhm=scl, dim=b0.dim())
    h = spatial.smooth(h, fwhm=scl, dim=b0.dim(), fn=torch.square)
    g = dist_push(g, dphi, None, fmap)
    h = dist_push(h, dphi, None, fmap)

    g *= 2*math.pi*fmap['delta_te']
    h *= (2*math.pi*fmap['delta_te']) ** 2

    fmap['badnwidth'] = bw0
    return l, g, h


def derivatives_dist_spgr(spgr, fmaps, theta, prm):
    """Distortion derivatives for a Spoiled Gradient Echo signal"""
    device = dict(device=prm.get('device', None))
    rho, r1, r2, b0, b0smo = theta['dat']
    rho, r1, r2, b0 = rho.exp(), r1.exp(), r2.exp(), b0 + b0smo
    e1 = r1.mul(-spgr['tr']).exp_()

    # compute signal at TE=0
    s0 = rho
    if 'b1-' in fmaps:
        s0 *= load(fmaps['b1-'], **device)
    if 'b1+' in fmaps:
        fa = load(fmaps['b1+'], **device) * spgr['fa']
        c, s = fa.cos(), fa.sin_()
        ce1 = c.mul_(e1)
    else:
        fa = spgr['fa']
        c, s = math.cos(fa), math.sin(fa)
        ce1 = c * e1
    s0 *= s

    @torch.jit.script
    def make_s0(s0, e1, ce1):
        return s0.mul_((1 - e1)/(1 - ce1))

    s0 = make_s0(s0, e1, ce1).abs_()

    # gradient and hessian of signal
    g = s0.new_zeros(s0.shape)
    h = s0.new_zeros(s0.shape)

    phi_pos, phi_neg, jac_pos, jac_neg = make_distortion(b0, spgr, prm)
    l = 0
    for e, te in enumerate(spgr['te']):

        # compute R2* decay
        s = b0.abs().add_(r2).mul_(-te).exp_().mul_(s0)

        # distort and residuals
        phi = phi_neg if (e % 2) else phi_pos                   # distortion
        jac = jac_neg if (e % 2) else jac_pos                   # pile-up
        s, gs = dist_pull(s, phi, jac, spgr, grad=True)         # distorted fit
        d = load(spgr['dat'][e], **device)                      # obs
        m = torch.isfinite(d).bitwise_not_()                    # mask
        m = m.bitwise_or_(d <= 0)                               # mask
        r = s.sub(d).masked_fill_(m, 0)                         # residuals
        l += 0.5 * ssq(r)                                       # log-likelihood
        del d

        m = dist_push(m.bitwise_not_(), phi, jac, spgr).mul_(s)
        a = dist_push(r.abs(), phi, jac, spgr).mul_(s)

        h.add_(a, alpha=te*te)                          # from R2*
        h.addcmul_(m, s, alpha=te*te)                   # from R2*
        h.addcmul_(gs, gs)                              # from distortion
        if prm.get('pileup', True):
            h.add_(div2(s, spgr))                       # from pileup
        del a

        # gradient
        isinv = 1
        if prm.get('polarity', -1) > 0 and (e % 2):
            isinv = -1
        elif prm.get('polarity', -1) < 0 and ((e+1) % 2) == 0:
            isinv = -1
        g.addcmul_(r, gs, value=isinv)                  # from distortion
        if prm.get('pileup', True):
            g.add_(div(r, spgr), value=isinv)           # from pileup
        r = dist_push(r, phi, jac, spgr).mul_(s)
        g.addcmul_(r, b0.sign(), value=-te)             # from R2*
        del r

    # modulate by noise precision
    sigma = spgr['sigma']
    isigma2 = 1 / (sigma*sigma)
    l *= isigma2
    g *= isigma2
    h *= isigma2

    return l, g, h


def derivatives_dist_se(se, fmaps, theta, prm):
    """Distortion derivatives for a Spin Echo signal"""
    device = dict(device=prm.get('device', None))
    rho, r1, r2, b0, b0smo = theta['dat']
    rho, r1, r2, b0 = rho.exp(), r1.exp(), r2.exp(), b0 + b0smo
    e1 = r1.mul(-se['tr']).exp_()

    # compute signal at TE=0
    s0 = rho
    if 'b1-' in fmaps:
        s0 *= load(fmaps['b1-'], **device)

    @torch.jit.script
    def make_s0(s0, e1):
        return s0.mul_(1 - e1)

    s0 = make_s0(s0, e1).abs_()

    # gradient and hessian of signal
    g = s0.new_zeros(s0.shape)
    h = s0.new_zeros(s0.shape)

    phi_pos, phi_neg, jac_pos, jac_neg = make_distortion(b0, se, prm)
    l = 0
    for e, te in enumerate(se['te']):

        # compute R2* decay
        s = r2.mul(-te).exp_().mul_(s0)

        # distort and residuals
        phi = phi_neg if (e % 2) else phi_pos                   # distortion
        jac = jac_neg if (e % 2) else jac_pos                   # pile-up
        s, gs = dist_pull(s, phi, jac, se, grad=True)           # distorted fit
        d = load(se['dat'][e], **device)                        # obs
        m = torch.isfinite(d).bitwise_not_()                    # mask
        m = m.bitwise_or_(d <= 0)                               # mask
        r = s.sub(d).masked_fill_(m, 0)                         # residuals
        l += 0.5 * ssq(r)                                       # log-likelihood
        del d

        h.addcmul_(gs, gs)                              # from distortion
        if prm.get('pileup', True):
            h.add_(div2(s, se))                         # from pileup

        # gradient
        isinv = 1
        if prm.get('polarity', -1) > 0 and (e % 2):
            isinv = -1
        elif prm.get('polarity', -1) < 0 and ((e+1) % 2) == 0:
            isinv = -1
        g.addcmul_(r, gs, value=isinv)                  # from distortion
        if prm.get('pileup', True):
            g.add_(div(r, se), value=isinv)             # from pileup
        del r

    # modulate by noise precision
    sigma = se['sigma']
    isigma2 = 1 / (sigma*sigma)
    l *= isigma2
    g *= isigma2
    h *= isigma2

    return l, g, h


def update_tv(theta, wrls, lam, prm):
    joint = prm.get('joint', True)
    lam_tv = [lam[k] for k in theta['indices'][:-1]]
    theta = theta['dat'][:-1]
    w, tv = spatial.membrane_weights(theta, factor=lam_tv, return_sum=True,
                                     joint=joint)
    wrls.copy_(w)
    return tv


@torch.jit.script
def ssq(x):
    return x.flatten().dot(x.flatten())


@torch.jit.script
def dot(x, y):
    return x.flatten().dot(y.flatten())


@torch.jit.script
def masknans_(x):
    x.masked_fill_(torch.isfinite(x).bitwise_not_(), 0)
    return x


def make_distortion(b0, acq, prm):
    """Prepare distortion sampling grids

    Parameters
    ----------
    b0 : (..., *spatial) tensor
    acq : dict(readout=int, polarity={+1, -1}),
          default={'readout': -1, 'polarity': +1}
    prm : dict(pileup=bool), default={'pileup': True}

    Returns
    -------
    phi_even, phi_odd, jac_even, jac_odd

    """

    if not prm.get('distortion', True):
        return (None,) * 4

    pileup = prm.get('pileup', True)
    dir = acq.get('readout', -1)
    polarity = acq.get('polarity', +1)
    bandwidth = acq.get('bandwidth', 1)

    if bandwidth is not 1:
        b0 = b0 / bandwidth
    else:
        b0 = b0.clone()

    if pileup:
        jac = spatial.diff1d(b0, dim=dir)
        ijac = 1 - jac
        jac = 1 + jac
    else:
        jac = ijac = None

    b0 = utils.movedim(b0, dir, -1)
    iphi = spatial.add_identity_grid_(-b0.unsqueeze(-1)).squeeze(-1)
    phi = spatial.add_identity_grid_(b0.unsqueeze(-1)).squeeze(-1)
    phi = utils.movedim(phi, -1, dir)
    iphi = utils.movedim(iphi, -1, dir)

    if polarity > 0:
        return phi, iphi, jac, ijac
    else:
        return iphi, phi, ijac, jac


def dist_pull(img, grid, jac, acq, grad=False, **kwargs):
    """Distortion + pile-up along a single dimension

    Parameters
    ----------
    img : (K, *spatial) tensor, Image
    grid : (*spatial) tensor, Sampling grid
    jac : (*spatial) tensor, Jacobian
    acq : dict(readout=int), default={'readout': -1}
    grad : bool

    Returns
    -------
    warped_img : (K, *spatial) tensor, Warped [+ moduled] image
    warped_grad : (K, *spatial) tensor, Warped [+  modulated] gradients

    """
    if grid is None:
        return (img, None) if grad else img
    dim = acq.get('readout', -1)
    kwargs.setdefault('extrapolate', True)
    kwargs.setdefault('bound', 'dft')
    img, grid = utils.movedim(img, dim, -1), utils.movedim(grid, dim, -1)
    img, grid = img.unsqueeze(-2), grid.unsqueeze(-1)
    warped = spatial.grid_pull(img, grid, **kwargs).squeeze(-2)
    if grad:
        wgrad = spatial.grid_grad(img, grid, **kwargs).squeeze(-1).squeeze(-2)
        wgrad = utils.movedim(wgrad, -1, dim)
    warped = utils.movedim(warped, -1, dim)
    if jac is not None:
        warped *= jac
        if grad:
            wgrad *= jac
    return (warped, wgrad) if grad else warped


def dist_push(img, grid, jac, acq, **kwargs):
    """Push an image by a transform along the last dimension

    This is the adjoint of `pull1d`.

    Parameters
    ----------
    img : (K, *spatial) tensor, Image
    grid : (*spatial) tensor, Sampling grid

    Returns
    -------
    pushed_img : (K, *spatial) tensor

    """
    if grid is None:
        return img
    dim = acq.get('readout', -1)
    kwargs.setdefault('extrapolate', True)
    kwargs.setdefault('bound', 'dft')
    if jac is not None:
        img = img * jac
    img, grid = utils.movedim(img, dim, -1), utils.movedim(grid, dim, -1)
    img, grid = img.unsqueeze(-2), grid.unsqueeze(-1)
    pushed = spatial.grid_push(img.to(grid), grid, **kwargs).squeeze(-2)
    pushed = utils.movedim(pushed, -1, dim)
    return pushed


def div(b0, acq):
    dir = acq.get('readout', -1)
    jac = spatial.diff1d(b0, dim=dir)
    return jac


def div2(b0, acq):
    dir = acq.get('readout', -1)
    bandwidth = acq.get('bandwidth', 1)

    b0 = b0.abs()
    b0 = utils.movedim(b0, dir, 0)
    d = torch.zeros_like(b0)
    d[:-1] += b0[1:]
    d[1:] += b0[:-1]
    d[-1] += b0[0]
    d[0] += b0[-1]
    d /= (2 * bandwidth) ** 2
    d = utils.movedim(d, 0, dir)
    return d