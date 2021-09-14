import torch
from nitorch import core, spatial
from ._options import ESTATICSOptions
from ._preproc import preproc, postproc
from ._utils import (hessian_loaddiag, hessian_matmul, hessian_solve,
                     smart_grid, smart_pull, smart_push, smart_grad,)
from ..utils import rls_maj
from nitorch.tools.qmri.param import ParameterMap, SVFDeformation, DenseDeformation
from nitorch.spatial import solve_grid_sym


def nonlin(data, opt=None):
    """Fit the ESTATICS model to multi-echo Gradient-Echo data.

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
        Observed GRE data.
    opt : Options, optional
        Algorithm options.

    Returns
    -------
    intecepts : sequence[GradientEcho]
        Echo series extrapolated to TE=0
    decay : estatics.ParameterMap
        R2* decay map
    distortions : sequence[ParameterizedDeformation], if opt.distortion.enable
        B0-induced distortion fields

    """

    opt = ESTATICSOptions().update(opt)
    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    # --- be polite ---
    print(f'Fitting a (multi) exponential decay model with {len(data)} contrasts. Echo times:')
    for i, contrast in enumerate(data):
        print(f'    - contrast {i:2d}: [' + ', '.join([f'{te*1e3:.1f}' for te in contrast.te]) + '] ms')

    # --- estimate noise / register / initialize maps ---
    data, maps, dist = preproc(data, opt)
    vx = spatial.voxel_size(maps.affine)

    # --- prepare regularization factor ---
    lam = opt.regularization.factor
    lam = core.py.make_list(lam)
    if len(lam) > 1:
        *lam, lam_decay = lam
    else:
        lam_decay = lam[0]
    lam = core.py.make_list(lam, len(maps)-1)
    lam.append(lam_decay)

    distprm = dict(
        factor=opt.distortion.factor,
        absolute=opt.distortion.absolute,
        membrane=opt.distortion.membrane,
        bending=opt.distortion.bending)

    # --- initialize weights (RLS) ---
    if (not opt.regularization.norm or
        opt.regularization.norm.lower() == 'none' or
        all(l == 0 for l in lam)):
        opt.regularization.norm = ''
    opt.regularization.norm = opt.regularization.norm.lower()
    mean_shape = maps.decay.volume.shape
    rls = None
    sumrls = 0
    if opt.regularization.norm in ('tv', 'jtv'):
        rls_shape = mean_shape
        if opt.regularization.norm == 'tv':
            rls_shape = (len(maps),) + rls_shape
        rls = ParameterMap(rls_shape, fill=1, **backend).volume
        sumrls = 0.5 * rls.sum(dtype=torch.double)

    if opt.regularization.norm:
        print(f'With {opt.regularization.norm.upper()} regularization:')
        print('    - log intercepts: [' + ', '.join([f'{i:.3g}' for i in lam[:-1]]) + ']')
        print(f'    - decay:          {lam[-1]:.3g}')
    else:
        print('Without regularization:')

    # --- compute derivatives ---
    grad = torch.empty((len(data) + 1,) + mean_shape, **backend)
    hess = torch.empty((len(data)*2 + 1,) + mean_shape, **backend)

    if opt.regularization.norm not in ('tv', 'jtv'):
        # no reweighting -> do more gauss-newton updates instead
        opt.optim.max_iter_gn *= opt.optim.max_iter_rls
        opt.optim.max_iter_rls = 1

    if opt.verbose:
        pstr = f'{"rls":^3s} | {"gn":^3s} | ' 
        if opt.distortion.enable:
            pstr += f'{"step":^4s} | ' 
        pstr += f'{"fit":^12s} + {"reg":^12s} + {"rls":^12s} '
        if opt.distortion.enable:
            pstr += f'+ {"dist":^12s} '
        pstr += f'= {"crit":^12s}'
        print(pstr)
        print('-' * len(pstr))

    ll_rls = []
    ll_max = -float('inf')
    crit = crit_prev = float('inf')
    sumrls_prev = sumrls
    reg = reg_prev = 0
    vreg = vreg_prev = 0
    lm = 1.  # levenberg marquardt
    lm_confidence = 0
    armijo_prev = 1

    for n_iter_rls in range(opt.optim.max_iter_rls):

        multi_rls = rls if opt.regularization.norm == 'tv' \
                    else [rls] * len(maps)

        ll_gn = []
        for n_iter_gn in range(opt.optim.max_iter_gn):

            # -----------------
            #    Update maps
            # -----------------

            decay = maps.decay
            crit_prev = crit
            crit = 0
            grad.zero_()
            hess.zero_()

            # --- loop over contrasts ---
            for i, (contrast, intercept, distortion) in enumerate(zip(data, maps.intercepts, dist)):
                # compute gradient
                crit1, g1, h1 = _nonlin_gradient(contrast, distortion,
                                                 intercept, decay, opt)

                # increment
                gind = [i, -1]
                grad[gind, ...] += g1
                hind = [2*i, -1, 2*i+1]
                hess[hind, ...] += h1
                crit += crit1

            # --- regularization ---
            reg_prev = reg
            reg = 0.
            if opt.regularization.norm:
                for i, (map, weight, l) in enumerate(zip(maps, multi_rls, lam)):
                    if not l:
                        continue
                    reg1, g1 = _nonlin_reg(map.volume, vx, weight, l)
                    reg += reg1
                    grad[i] += g1

            # --- gauss-newton ---
            if not torch.isfinite(hess).all():
                print('WARNING: NaNs in hess')
            if opt.regularization.norm:
                hess = hessian_loaddiag(hess, 1e-6, 1e-8)
                deltas = _nonlin_solve(hess, grad, multi_rls, lam, vx, opt)
            else:
                hess = hessian_loaddiag(hess, 1e-6, 1e-8)
                deltas = hessian_solve(hess, grad)
            if not torch.isfinite(deltas).all():
                print('WARNING: NaNs in delta (non stable Hessian)')

            for map, delta in zip(maps, deltas):
                map.volume -= delta
                if map.min is not None or map.max is not None:
                    map.volume.clamp_(map.min, map.max)

            # ------------------------
            #    Update distortions
            # ------------------------
            n_total = n_iter_rls*opt.optim.max_iter_gn + n_iter_gn
            if opt.distortion.enable:  #  and n_total > 5:

                # --- intermediate ll (distortion) ---
                # This is the gain due to the distortion update since we
                # compute the criterion *before* computing the derivatives
                if crit + vreg <= crit_prev + vreg_prev:
                    evol = '<='
                else:
                    evol = '>'
                if opt.verbose:
                    ll_tmp = crit + reg_prev + vreg + sumrls
                    pstr = (f'{n_iter_rls:3d} | {n_iter_gn:3d} | {"dist":4s} | '
                            f'{crit:12.6g} + {reg_prev:12.6g} + {sumrls:12.6g} ')
                    if opt.distortion.enable:
                        pstr += f'+ {vreg:12.6g} '
                    pstr += f'= {ll_tmp:12.6g} | '
                    pstr += f'{evol}'
                    print(pstr)

                crit_prev = crit
                vreg_prev = vreg
                crit = 0
                vreg = 0
                # --- loop over contrasts ---
                for i, (contrast, intercept, distortion) in enumerate(zip(data, maps.intercepts, dist)):
                    crit1, g, h = _distortion_gradient(contrast, distortion, intercept, decay, opt)
                    crit += crit1

                    # --- regularization ---
                    def momentum(vol):
                        if contrast.readout is None:
                            g1 = spatial.regulariser_grid(vol, **distprm, bound='dft',
                                                          voxel_size=distortion.voxel_size)
                        else:
                            distprm1 = dict(distprm)
                            distprm1['factor'] *= distortion.voxel_size[contrast.readout] ** 2
                            g1 = spatial.regulariser(vol[None], **distprm1, dim=3, bound='dft',
                                                     voxel_size=distortion.voxel_size)[0]
                        return g1
                    
                    if contrast.readout is None:
                        vol = distortion.volume
                    else:
                        vol = distortion.volume[..., contrast.readout]
                    g1 = momentum(vol)
                    reg1 = vol.flatten().dot(g1.flatten())
                    vreg += 0.5 * reg1
                    g += g1
                    del g1

                    # --- gauss-newton ---
                    if not torch.isfinite(h).all():
                        print('WARNING: NaNs in hess')
                    if contrast.readout is None:
                        h = core.utils.movedim(h, -1, -4)
                        h = hessian_loaddiag(h, lm*1e-6, 1e-8)
                        h = core.utils.movedim(h, -4, -1)
                        distprm1 = dict(distprm)
                        distprm1['factor'] *= lm
                        delta = spatial.solve_grid_fmg(h, g, **distprm1, bound='dft',
                                                       voxel_size=distortion.voxel_size,
                                                       verbose=opt.verbose-1,
                                                       nb_iter=opt.optim.max_iter_cg,
                                                       tolerance=opt.optim.tolerance_cg)
                    else:
                        h = hessian_loaddiag(h[None], lm*1e-6, 1e-8)[0]
                        distprm1 = dict(distprm)
                        distprm1['factor'] *= (distortion.voxel_size[contrast.readout] ** 2)
                        distprm1['factor'] *= (1 + lm*1e-6)
                        delta = spatial.solve_field_fmg(h[None], g[None], **distprm1, dim=3,
                                                        bound='dft',
                                                        voxel_size=distortion.voxel_size,
                                                        nb_iter=opt.optim.max_iter_cg,
                                                        tolerance=opt.optim.tolerance_cg,
                                                        verbose=opt.verbose-1)[0]
                    if not torch.isfinite(delta).all():
                        print('WARNING: NaNs in delta (non stable Hessian)')
                            
                    # --- line search ---
                    armijo, armijo_prev = armijo_prev, 0
                    ok = False
                    dd = momentum(delta)
                    dv = dd.flatten().dot(vol.flatten())
                    dd = dd.flatten().dot(dd.flatten())
                    for _ in range(12):
                        vol.sub_(delta, alpha=(armijo - armijo_prev))
                        armijo_prev = armijo
                        new_vreg = 0.5 * armijo*(armijo*dd - 2 * dv)
                        new_crit = _nonlin_gradient(contrast, distortion, intercept, decay, opt, do_grad=False)
                        # print(f'{new_crit.item():12.6g}, {new_vreg.item():12.6g}, '
                        #       f'{new_crit.item() + new_vreg.item():12.6g}, {crit1.item():12.6g}')
                        if new_crit + new_vreg <= crit1:
                            # print(f'armijo {armijo} -> :D')
                            ok = True
                            break
                        else:
                            # print(f'armijo {armijo} -> :(')
                            armijo = armijo / 2
                    if not ok:
                        vol.add_(delta, alpha=armijo_prev)
                    armijo_prev = 1
                    del delta, g, h

                # --- intermediate ll (param) ---
                # This is the gain due to the distortion update since we
                # compute the criterion *before* computing the derivatives
                if crit + reg <= crit_prev + reg_prev:
                    evol = '<='
                else:
                    evol = '>'
                if opt.verbose:
                    ll_tmp = crit + reg_prev + vreg + sumrls
                    pstr = (f'{n_iter_rls:3d} | {n_iter_gn:3d} | {"prm":4s} | '
                            f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} ')
                    pstr += f'+ {vreg_prev:12.6g} '
                    pstr += f'= {ll_tmp:12.6g} | '
                    pstr += f'{evol}'
                    print(pstr)

            # ------------------
            #    Compute gain
            # ------------------
            ll = crit + reg + vreg + sumrls
            ll_max = max(ll_max, ll)
            ll_prev = ll_gn[-1] if ll_gn else float('inf')
            gain = (ll_prev - ll) / (ll_max - ll_prev)
            ll_gn.append(ll)
            if opt.verbose:
                pstr = f'{n_iter_rls:3d} | {n_iter_gn:3d} | '
                if opt.distortion.enable:
                    pstr += f'{"----":4s} | '
                pstr += f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} '
                if opt.distortion.enable:
                    pstr += f'+ {vreg:12.6g} '
                pstr += f'= {ll:12.6g} | gain = {gain:7.2g}'
                print(pstr)
                _show_maps(maps, dist)
            if gain < opt.optim.tolerance_gn:
                break

        # --- Update RLS weights ---
        if opt.regularization.norm in ('tv', 'jtv'):
            rls = _nonlin_rls(maps, lam, opt.regularization.norm)
            sumrls = 0.5 * rls.sum(dtype=torch.double)
            eps = core.constants.eps(rls.dtype)
            rls = rls.clamp_min_(eps).reciprocal_()

            # --- Compute gain ---
            # (we are late by one full RLS iteration when computing the
            #  gain but we save some computations)
            ll = ll_gn[-1]
            ll_prev = ll_rls[-1][-1] if ll_rls else float('inf')
            ll_rls.append(ll_gn)
            gain = (ll_prev - ll) / (ll_max - ll_prev)
            if gain < opt.optim.tolerance_rls:
                print(f'Converged ({gain:7.2g})')
                break

    # --- Prepare output ---
    out = postproc(maps, data)
    if opt.distortion.enable:
        out = (*out, dist)
    return out


@torch.jit.script
def recon_fit(inter, slope, te: float):
    return inter.add(slope, alpha=-te).exp()


@torch.jit.script
def ssq(x):
    return x.square().sum(dtype=torch.double)


@torch.jit.script
def get_mask_missing(dat, fit):
    return ~(torch.isfinite(fit) & torch.isfinite(dat) & (dat > 0))


@torch.jit.script
def mask_nan_(x, value: float = 0.):
    return x.masked_fill_(~torch.isfinite(x), value)

    
def _nonlin_gradient(contrast, distortion, intercept, decay, opt, do_grad=True):
    """Compute the gradient and Hessian of the parameter maps with
    respect to one contrast.

    Parameters
    ----------
    contrast : (nb_echo, *obs_shape) GradientEchoMulti
        A single echo series (with the same weighting)
    distortion : ParameterizedDeformation
        A model of distortions caused by B0 inhomogeneities.
    intercept : (*recon_shape) ParameterMap
        Log-intercept of the contrast
    decay : (*recon_shape) ParameterMap
        Exponential decay
    opt : Options
    do_grad : bool, default=True

    Returns
    -------
    crit : () tensor
        Log-likelihood
    grad : (2, *recon_shape) tensor, if `do_grad`
        Gradient with respect to:
            [0] intercept
            [1] decay
    hess : (3, *recon_shape) tensor, if `do_grad`
        Hessian with respect to:
            [0] intercept ** 2
            [1] decay ** 2
            [2] intercept * decay

    """

    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    obs_shape = contrast.volume.shape[1:]
    recon_shape = intercept.volume.shape
    aff = core.linalg.lmdiv(intercept.affine, contrast.affine)
    aff = aff.to(**backend)
    lam = 1/contrast.noise

    # pull parameter maps to observed space
    grid = smart_grid(aff, obs_shape, recon_shape)
    inter = smart_pull(intercept.fdata(**backend)[None], grid)[0]
    slope = smart_pull(decay.fdata(**backend)[None], grid)[0]
    if distortion:
        grid_up, grid_down = distortion.exp2(add_identity=True)
    else:
        grid_up = grid_down = None
        
    crit = 0
    grad = torch.zeros((2,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((3,) + obs_shape, **backend) if do_grad else None

    for e, echo in enumerate(contrast):

        te = echo.te
        blip = echo.blip or (2*(e % 2) - 1)
        grid_blip = grid_up if blip > 0 else grid_down

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)  # observed
        dat = mask_nan_(dat)
        msk = dat < 1
        fit = recon_fit(inter, slope, te)                    # fitted
        pull_fit = smart_pull(fit[None], grid_blip, bound='dft')[0]
        # msk = get_mask_missing(dat, pull_fit)                # mask of observed
        # dat.masked_fill_(msk, 0)
        pull_fit.masked_fill_(msk, 0)
        res = dat.neg_().add_(pull_fit)
        del dat, msk, pull_fit

        # compute log-likelihood
        crit = crit + 0.5 * lam * ssq(res)

        if do_grad:
            if grid_blip is not None:
                res0 = res
                res = smart_push(res0[None], grid_blip, bound='dft')[0]
                abs_res = smart_push(res0.abs_()[None], grid_blip, bound='dft')[0]
                abs_res.mul_(fit)
                del res0
            
            # compute gradient and Hessian in observed space
            #
            #   grad[inter]       =           lam * res * fit
            #   grad[decay]       =     -te * lam * res * fit
            #   hess[inter**2]    =           lam * fit * fit + abs(grad[inter])
            #   hess[decay**2]    = (te*te) * lam * fit * fit + abs(grad[decay])
            #   hess[inter*decay] =     -te * lam * fit * fit
            #
            # I tried to put that into an "accumulation" function but it 
            # does super weird stuff, so I keep it in the main loop. I am 
            # saving allocations here so I think it's faster than 
            # torchscript.
            
            res.mul_(fit)
            grad[0].add_(res, alpha=lam)
            grad[1].add_(res, alpha=-te*lam)
            if grid_blip is None:
                abs_res = res.abs_()
            fit2 = fit.mul_(fit)
            hess[2].add_(fit2, alpha=-te*lam)
            fit2.add_(abs_res)
            hess[0].add_(fit2, alpha=lam)
            hess[1].add_(fit2, alpha=lam*(te*te))
            
            del res, fit, abs_res, fit2

    if not do_grad:
        return crit
            
    mask_nan_(grad)
    mask_nan_(hess[:-1], 1e-3)  # diagonal
    mask_nan_(hess[-1])         # off-diagonal

    # push gradient and Hessian to recon space
    grad = smart_push(grad, grid, recon_shape)
    hess = smart_push(hess, grid, recon_shape)
    return crit, grad, hess


def _distortion_gradient(contrast, distortion, intercept, decay, opt, do_grad=True):
    """Compute the gradient and Hessian of the distortion field.

    Parameters
    ----------
    contrast : (nb_echo, *obs_shape) GradientEchoMulti
        A single echo series (with the same weighting)
    distortion : ParameterizedDeformation
        A model of distortions caused by B0 inhomogeneities.
    intercept : (*recon_shape) ParameterMap
        Log-intercept of the contrast
    decay : (*recon_shape) ParameterMap
        Exponential decay
    opt : Options

    Returns
    -------
    crit : () tensor
        Log-likelihood
    grad : (*shape, 3) tensor
    hess : (*shape, 6) tensor

    """

    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    obs_shape = contrast.volume.shape[1:]
    recon_shape = intercept.volume.shape
    aff = core.linalg.lmdiv(intercept.affine, contrast.affine)
    aff = aff.to(**backend)
    lam = 1/contrast.noise

    # pull parameter maps to observed space
    grid = smart_grid(aff, obs_shape, recon_shape)
    inter_slope = torch.stack([intercept.fdata(**backend),
                               decay.fdata(**backend)])
    inter, slope = smart_pull(inter_slope, grid)
    del inter_slope
    readout = contrast.readout
    grid_up, grid_down = distortion.exp2(add_identity=True)

    crit = 0
    grad = torch.zeros(obs_shape + (3,), **backend) if do_grad else None
    hess = torch.zeros(obs_shape + (6,), **backend) if do_grad else None

    for e, echo in enumerate(contrast):

        blip = echo.blip or (2*(e % 2) - 1)
        grid_blip = grid_up if blip > 0 else grid_down

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)  # observed
        dat = mask_nan_(dat)
        msk = dat < 1
        fit = recon_fit(inter, slope, echo.te)               # fitted
        if do_grad and isinstance(distortion, DenseDeformation):
            if not (grid_blip == 0).any():
                gfit = spatial.diff(fit, bound='dft', dim=[-3, -2, -1])
            else:
                gfit = smart_grad(fit[None], grid_blip, bound='dft')[0]
        fit = smart_pull(fit[None], grid_blip, bound='dft')[0]   # fitted o phi
        if do_grad and isinstance(distortion, SVFDeformation):
            gfit = spatial.diff(fit, bound='dft', dim=[-3, -2, -1])  # D(fitted o phi)
        # msk = get_mask_missing(dat, fit)    # mask of missing values
        # dat.masked_fill_(msk, 0)
        fit.masked_fill_(msk, 0)
        gfit.masked_fill_(msk.unsqueeze(-1), 0)
        res = dat.neg_().add_(fit)
        del dat, fit, msk

        # compute log-likelihood
        crit = crit + 0.5 * lam * ssq(res)

        if do_grad:
            g1 = res.unsqueeze(-1).mul(gfit)
            h1 = torch.zeros_like(hess)
            if readout is None:
                h1[..., :3] = gfit.square()
                h1[..., 3] = gfit[..., 0] * gfit[..., 1]
                h1[..., 4] = gfit[..., 0] * gfit[..., 2]
                h1[..., 5] = gfit[..., 1] * gfit[..., 2]
            else:
                h1[..., readout] = gfit[..., readout].square()

            # propagate backward
            if isinstance(distortion, SVFDeformation):
                vel = distortion.volume
                if blip < 0:
                    vel = -vel
                g1, h1 = spatial.exp_backward(vel, g1, h1, steps=distortion.steps)
            if blip < 0:
                g1 = g1.neg_()

            grad.add_(g1, alpha=lam)
            hess.add_(h1, alpha=lam)

    if not do_grad:
        return crit

    if readout is None:
        mask_nan_(grad)
        mask_nan_(hess[:-3], 1e-3)  # diagonal
        mask_nan_(hess[-3:])        # off-diagonal
    else:
        grad = grad[..., readout]
        hess = hess[..., readout]
        mask_nan_(grad)
        mask_nan_(hess)

    return crit, grad, hess


def _nonlin_reg(map, vx=1., rls=None, lam=1., do_grad=True):
    """Compute the gradient of the regularisation term.

    The regularisation term has the form:
    `0.5 * lam * sum(w[i] * (g+[i]**2 + g-[i]**2) / 2)`
    where `i` indexes a voxel, `lam` is the regularisation factor,
    `w[i]` is the RLS weight, `g+` and `g-` are the forward and
    backward spatial gradients of the parameter map.

    Parameters
    ----------
    map : (*shape) tensor
        Parameter map
    vx : float or sequence[float]
        Voxel size
    rls : (*shape) tensor, optional
        Weights from the reweighted least squares scheme
    lam : float, default=1
        Regularisation factor
    do_grad : bool, default=True
        Return both the criterion and gradient

    Returns
    -------
    reg : () tensor
        Regularisation term
    grad : (*shape) tensor
        Gradient with respect to the parameter map

    """
    grad = spatial.membrane(map, weights=rls, dim=3, voxel_size=vx).mul_(lam)

    # grad_fwd = spatial.diff(map, dim=[0, 1, 2], voxel_size=vx, side='f')
    # grad_bwd = spatial.diff(map, dim=[0, 1, 2], voxel_size=vx, side='b')
    # if rls is not None:
    #     grad_fwd *= rls[..., None]
    #     grad_bwd *= rls[..., None]
    # grad_fwd = spatial.div(grad_fwd, dim=[0, 1, 2], voxel_size=vx, side='f')
    # grad_bwd = spatial.div(grad_bwd, dim=[0, 1, 2], voxel_size=vx, side='b')
    #
    # grad = grad_fwd
    # grad += grad_bwd
    # grad *= lam / 2   # average across side (2)

    if do_grad:
        reg = (map * grad).sum(dtype=torch.double)
        return 0.5 * reg, grad
    else:
        grad *= map
        return 0.5 * grad.sum(dtype=torch.double)


def _nonlin_solve(hess, grad, rls, lam, vx, opt):
    """Solve the regularized linear system

    Parameters
    ----------
    hess : (2*P+1, *shape) tensor
    grad : (P+1, *shape) tensor
    rls : (P+1, *shape) tensor_like
    lam : (P,) sequence[float]
    vx : (D,) sequence[float]
    opt : Options

    Returns
    -------
    delta : (P+1, *shape) tensor

    """
    def matvec(m, x):
        m = m.transpose(-1, -4)
        x = x.transpose(-1, -4)
        return hessian_matmul(m, x).transpose(-4, -1)

    def matsolve(m, x):
        m = m.transpose(-1, -4)
        x = x.transpose(-1, -4)
        return hessian_solve(m, x).transpose(-4, -1)

    def matdiag(m, d):
        return m[..., ::2]

    rls = torch.stack(rls)
    return spatial.solve_field_fmg(hess, grad, rls, factor=lam, membrane=1,
                                   voxel_size=vx, verbose=opt.verbose - 1,
                                   nb_iter=opt.optim.max_iter_cg,
                                   tolerance = opt.optim.tolerance_cg,
                                   matvec=matvec, matsolve=matsolve, matdiag=matdiag)

    def hess_fn(x):
        result = hessian_matmul(hess, x)
        if not opt.regularization.norm:
            return result
        for i, (map, weight, l) in enumerate(zip(x, rls, lam)):
            if not l:
                continue
            _, res1 = _nonlin_reg(map, vx, weight, l)
            result[i] += res1
        return result

    # The Hessian is A = H + L, where H corresponds to the data term
    # and L to the regularizer. Note that, L = D'WD where D is the
    # gradient operator, D' the divergence and W a diagonal matrix
    # that contains the RLS weights.
    # We use (H + diag(|D'D|w)) as a preconditioner because it is easy to
    # invert and majorises the true Hessian.
    hessp = hess.clone()
    smo = torch.as_tensor(vx).square().reciprocal().sum().item()
    for i, (weight, l) in enumerate(zip(rls, lam)):
        if not l:
            continue
        hessp[2*i] += l * (rls_maj(weight, vx) if weight is not None else 4*smo)

    def precond(x):
        return hessian_solve(hessp, x)

    result = core.optim.cg(hess_fn, grad, precond=precond,
                           verbose=(opt.verbose > 1), stop='norm',
                           max_iter=opt.optim.max_iter_cg,
                           tolerance=opt.optim.tolerance_cg)
    return result


def _nonlin_rls(maps, lam=1., norm='jtv'):
    """Update the (L1) weights.

    Parameters
    ----------
    map : (P, *shape) ParameterMaps
        Parameter map
    lam : float or (P,) sequence[float], default=1
        Regularisation factor
    norm : {'tv', 'jtv'}, default='jtv'

    Returns
    -------
    rls : ([P], *shape) tensor
        Weights from the reweighted least squares scheme
    """

    if norm not in ('tv', 'jtv', '__internal__'):
        return None

    if isinstance(maps, ParameterMap):
        # single map
        # this should only be an internal call
        # -> we return the squared gradient map
        assert norm == '__internal__'
        vx = spatial.voxel_size(maps.affine)
        grad_fwd = spatial.diff(maps.volume, dim=[0, 1, 2], voxel_size=vx, side='f')
        grad_bwd = spatial.diff(maps.volume, dim=[0, 1, 2], voxel_size=vx, side='b')

        grad = grad_fwd.square_().sum(-1)
        grad += grad_bwd.square_().sum(-1)
        grad *= lam / 2   # average across side (2)
        return grad

    # multiple maps

    if norm == 'tv':
        rls = []
        for map, l in zip(maps, lam):
            rls1 = _nonlin_rls(map, l, '__internal__')
            rls1 = rls1.sqrt_()
            rls.append(rls1)
    else:
        assert norm == 'jtv'
        rls = 0
        for map, l in zip(maps, lam):
            rls += _nonlin_rls(map, l, '__internal__')
        rls = rls.sqrt_()

    return rls


def _show_maps(maps, dist):
    import matplotlib.pyplot as plt
    has_dist = any([d is not None for d in dist])
    ncol = max(len(maps), len(dist))
    for i, map in enumerate(maps):
        plt.subplot(1 + has_dist, ncol, i+1)
        plt.imshow(map.volume[:, :, map.shape[-1]//2].cpu())
        plt.axis('off')
        plt.colorbar()
    if has_dist:
        for i, dst in enumerate(dist):
            if dst is None:
                continue
            vol = dst.volume
            plt.subplot(1 + has_dist, ncol, i+1+ncol)
            plt.imshow(vol[:, :, dst.shape[-2]//2, :].square().sum(-1).sqrt().cpu())
            plt.axis('off')
            plt.colorbar()
    plt.show()
