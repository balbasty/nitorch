import torch
from nitorch import core, spatial
from ._options import ESTATICSOptions
from ._preproc import preproc, postproc
from ._utils import (hessian_loaddiag_, hessian_matmul, hessian_solve,
                     smart_grid, smart_pull, smart_push, smart_grad, )
from ..utils import rls_maj
from nitorch.spatial import solve_grid_sym
from nitorch.tools.qmri.param import ParameterMap, SVFDeformation, DenseDeformation
from typing import Optional


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

    # --- options ------------------------------------------------------
    # we deepcopy all options so that we can overwrite/simplify them in place
    opt = ESTATICSOptions().update(opt).cleanup_()
    backend = dict(dtype=opt.backend.dtype, device=opt.backend.device)

    # --- be polite ----------------------------------------------------
    print(f'Fitting a (multi) exponential decay model with {len(data)} contrasts. Echo times:')
    for i, contrast in enumerate(data):
        print(f'    - contrast {i:2d}: [' + ', '.join([f'{te*1e3:.1f}' for te in contrast.te]) + '] ms')

    # --- estimate noise / register / initialize maps ------------------
    data, maps, dist = preproc(data, opt)
    vx = spatial.voxel_size(maps.affine)
    nb_contrasts = len(maps) - 1

    # --- prepare regularization factor --------------------------------
    # 1. Parameter maps regularization
    #   -> we want lam = [*lam_intercepts, lam_decay]
    *lam, lam_decay = opt.regularization.factor
    lam = core.py.make_list(lam, nb_contrasts)
    lam.append(lam_decay)
    if not any(lam):
        opt.regularization.norm = ''
    # 2. Distortion fields regularization
    lam_dist = dict(
        factor=opt.distortion.factor,
        absolute=opt.distortion.absolute,
        membrane=opt.distortion.membrane,
        bending=opt.distortion.bending)

    # --- initialize weights (RLS) -------------------------------------
    iter_rls = make_iter_rls(nb_contrasts)
    mean_shape = maps.decay.volume.shape
    rls = None
    sumrls = 0
    if opt.regularization.norm.endswith('tv'):
        rls_shape = mean_shape
        if opt.regularization.norm == 'tv':
            rls_shape = (len(maps), *rls_shape)
        rls = ParameterMap(rls_shape, fill=1, **backend).volume
        sumrls = 0.5 * rls.sum(dtype=torch.double)

    if opt.regularization.norm:
        print('Regularization:')
        print(f'    - type:           {opt.regularization.norm.upper()}')
        print(f'    - log intercepts: [' + ', '.join([f'{i:.3g}' for i in lam[:-1]]) + ']')
        print(f'    - decay:          {lam[-1]:.3g}')
    else:
        print('Without regularization')

    if opt.distortion.enable:
        print('Distortion correction:')
        print(f'    - model:          {opt.distortion.model.lower()}')
        print(f'    - absolute:       {opt.distortion.absolute * opt.distortion.factor}')
        print(f'    - membrane:       {opt.distortion.membrane * opt.distortion.factor}')
        print(f'    - bending:        {opt.distortion.bending * opt.distortion.factor}')
        print(f'    - te_scaling:     {opt.distortion.te_scaling or "no"}')

    else:
        print('Without distortion correction')

    # --- initialize nb of iterations ----------------------------------
    if not opt.regularization.norm.endswith('tv'):
        # no reweighting -> do more gauss-newton updates instead
        opt.optim.max_iter_gn *= opt.optim.max_iter_rls
        opt.optim.max_iter_rls = 1
    print('Optimization:')
    if opt.regularization.norm.endswith('tv'):
        print(f'    - IRLS iterations: {opt.optim.max_iter_rls}'
              f' (tolerance: {opt.optim.tolerance_rls})')
    print(f'    - GN iterations:   {opt.optim.max_iter_gn}'
          f' (tolerance: {opt.optim.tolerance_gn})')
    print(f'    - FMG cycles:      2')
    print(f'    - CG iterations:   {opt.optim.max_iter_cg}'
          f' (tolerance: {opt.optim.tolerance_cg})')

    # ------------------------------------------------------------------
    #                     MAIN OPTIMIZATION LOOP
    # ------------------------------------------------------------------

    # --- allocate derivatives wrt parameter maps ----------------------
    grad = torch.empty((len(data) + 1,) + mean_shape, **backend)
    hess = torch.empty((len(data)*2 + 1,) + mean_shape, **backend)

    if opt.verbose:
        pstr = f'{"rls":^3s} | {"gn":^3s} | ' 
        if opt.distortion.enable:
            pstr += f'{"step":^4s} | ' 
        pstr += f'{"fit":^12s} + {"reg":^12s} + {"rls":^12s} '
        if opt.distortion.enable:
            pstr += f'+ {"dist":^12s} '
        pstr += f'= {"crit":^12s}'
        print('\n' + pstr)
        print('-' * len(pstr))

    # --- initialize tracking of the objective function ----------------
    ll_rls = []
    ll_max = -float('inf')
    crit = float('inf')
    vreg = 0
    reg = 0
    sumrls_prev = sumrls
    rls_changed = False
    armijo_dist_prev = [1] * nb_contrasts

    # --- RLS loop -----------------------------------------------------
    #   > max_iter_rls == 1 if regularization is not (J)TV
    for n_iter_rls in range(1, opt.optim.max_iter_rls+1):

        # --- Gauss-Newton ---------------------------------------------
        # . If distortion correction is enabled, we perform one step of
        #   parameter update and one step of distortion update per iteration.
        # . We keep track of the objective within each RLS iteration so
        #   that we can stop Gauss-Newton early if we it reaches a plateau.
        ll_gn = []
        for n_iter_gn in range(1, opt.optim.max_iter_gn+1):

            # ----------------------------------------------------------
            #    Update parameter maps
            # ----------------------------------------------------------

            crit_prev = crit
            reg_prev = reg

            # --- loop over contrasts ----------------------------------
            crit = 0
            grad.zero_()
            hess.zero_()
            for i, (contrast, intercept, distortion) \
                    in enumerate(zip(data, maps.intercepts, dist)):
                # compute gradient
                crit1, g1, h1 = derivatives_parameters(
                    contrast, distortion, intercept, maps.decay, opt)
                # increment
                gind = [i, -1]
                grad[gind, ...] += g1
                hind = [2*i, -1, 2*i+1]
                hess[hind, ...] += h1
                crit += crit1

            # --- regularization ---------------------------------------
            reg = 0.
            if opt.regularization.norm:
                for i, (map, weight, l) \
                        in enumerate(zip(maps, iter_rls(rls), lam)):
                    if not l:
                        continue
                    g1 = spatial.membrane(map.volume, weights=weight, dim=3,
                                          voxel_size=vx).mul_(l)
                    reg1 = 0.5 * dot(map.volume, g1)
                    reg += reg1
                    grad[i] += g1

            # --- track RLS improvement --------------------------------
            # Updating the RLS weights changes `sumrls` and `reg`. Now
            # that we have the updated value of `reg` (before it is
            # modified by the map update), we can check that updating the
            # weights indeed improved the objective.
            if opt.verbose and rls_changed:
                rls_changed = False
                if reg + sumrls <= reg_prev + sumrls_prev:
                    evol = '<='
                else:
                    evol = '>'
                ll_tmp = crit_prev + reg + vreg + sumrls
                pstr = (f'{n_iter_rls:3d} | {n_iter_gn:3d} | {"rls":4s} | '
                        f'{crit_prev:12.6g} + {reg:12.6g} + {sumrls:12.6g} ')
                if opt.distortion.enable:
                    pstr += f'+ {vreg:12.6g} '
                pstr += f'= {ll_tmp:12.6g} | '
                pstr += f'{evol}'
                print(pstr)

            # --- gauss-newton -----------------------------------------
            # Computing the GN step involves solving H\g
            hess = check_nans_(hess, warn='hessian')
            hess = hessian_loaddiag_(hess, 1e-6, 1e-8)
            deltas = hessian_solve(hess, grad)
            deltas = check_nans_(deltas, warn='delta')

            if not opt.distortion.enable:
                # No need for a line search
                for map, delta in zip(maps, deltas):
                    map.volume -= delta
                    if map.min is not None or map.max is not None:
                        map.volume.clamp_(map.min, map.max)

                # --- track general improvement ------------------------
                ll = crit + reg + sumrls
                ll_max = max(ll_max, ll)
                ll_prev = ll_gn[-1] if ll_gn else float('inf')
                gain = (ll_prev - ll) / (ll_max - ll_prev)
                ll_gn.append(ll)
                if opt.verbose:
                    pstr = f'{n_iter_rls:3d} | {n_iter_gn:3d} | '
                    pstr += f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} '
                    pstr += f'= {ll:12.6g} | gain = {gain:7.2g}'
                    print(pstr)
                    _show_maps(maps, dist)
                if gain < opt.optim.tolerance_gn:
                    break

            else:  # === distortion.enabled ============================

                # --- line search --------------------------------------
                # Block coordinate descent makes it a bit less robust
                # -> we use a line search
                dd = []
                if opt.regularization.norm:
                    for i, (delta, weight, l) \
                            in enumerate(zip(deltas, iter_rls(rls), lam)):
                        if not l:
                            continue
                        g1 = spatial.membrane(delta, weights=weight,
                                              dim=3, voxel_size=vx).mul_(l)
                        dd.append(g1)
                        del g1
                dv = sum([dd1.flatten().dot(map.volume.flatten())
                          for dd1, map in zip(dd, maps)])
                dd = sum([dd1.flatten().dot(delta.flatten())
                          for dd1, delta in zip(dd, deltas)])
                armijo, armijo_prev = 1, 0
                ok = False
                for n_ls in range(12):
                    for map, delta in zip(maps, deltas):
                        map.volume.sub_(delta, alpha=(armijo - armijo_prev))
                        # if map.min is not None or map.max is not None:
                        #     map.volume.clamp_(map.min, map.max)
                    armijo_prev = armijo
                    new_reg = 0.5 * armijo * (armijo * dd - 2 * dv)
                    new_crit = 0
                    for i, (contrast, intercept, distortion) \
                            in enumerate(zip(data, maps.intercepts, dist)):
                        # compute gradient
                        new_crit += derivatives_parameters(
                            contrast, distortion, intercept, maps.decay, opt,
                            do_grad=False)
                    if new_crit + new_reg <= crit:
                        ok = True
                        break
                    else:
                        armijo = armijo / 2
                if not ok:
                    for map, delta in zip(maps, deltas):
                        map.volume.add_(delta, alpha=armijo_prev)
                    new_crit = crit
                    new_reg = reg

                new_reg = reg + new_reg

                # ------------------------------------------------------
                #    Update distortions
                # ------------------------------------------------------

                # --- track parameter map improvement ------------------
                if not ok:
                    evol = '== (x)'
                elif new_crit + new_reg <= crit + reg:
                    evol = f'<= ({n_ls:2d})'
                else:
                    evol = f'> ({n_ls:2d})'
                crit = new_crit
                reg = new_reg
                if opt.verbose:
                    ll_tmp = crit + reg + vreg + sumrls
                    pstr = (f'{n_iter_rls:3d} | {n_iter_gn:3d} | {"prm":4s} | '
                            f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} ')
                    if opt.distortion.enable:
                        pstr += f'+ {vreg:12.6g} '
                    pstr += f'= {ll_tmp:12.6g} | '
                    pstr += f'{evol}'
                    print(pstr)

                crit = 0
                vreg = 0
                new_crit = 0
                new_vreg = 0
                # --- loop over contrasts ------------------------------
                for i, (contrast, intercept, distortion) \
                        in enumerate(zip(data, maps.intercepts, dist)):
                    crit1, g, h = derivatives_distortion(
                        contrast, distortion, intercept, maps.decay, opt)
                    crit += crit1

                    # --- regularization -------------------------------
                    def momentum(vol):
                        if contrast.readout is None:
                            g1 = spatial.regulariser_grid(
                                vol, **lam_dist, bound='dft',
                                voxel_size=distortion.voxel_size)
                        else:
                            distprm1 = dict(lam_dist)
                            distprm1['factor'] *= distortion.voxel_size[contrast.readout] ** 2
                            g1 = spatial.regulariser(
                                vol[None], **distprm1, dim=3, bound='dft',
                                voxel_size=distortion.voxel_size)[0]
                        return g1
                    
                    if contrast.readout is None:
                        vol = distortion.volume
                    else:
                        vol = distortion.volume[..., contrast.readout]
                    g1 = momentum(vol)
                    vreg1 = 0.5 * vol.flatten().dot(g1.flatten())
                    vreg += vreg1
                    g += g1
                    del g1

                    # --- gauss-newton ---------------------------------
                    h = check_nans_(h, warn='hessian (distortion)')
                    if contrast.readout is None:
                        h = core.utils.movedim(h, -1, -4)
                        h = hessian_loaddiag_(h, 1e-6, 1e-8)
                        h = core.utils.movedim(h, -4, -1)
                        delta = spatial.solve_grid_fmg(
                            h, g, **lam_dist, bound='dft',
                            voxel_size=distortion.voxel_size,
                            verbose=opt.verbose-1,
                            nb_iter=opt.optim.max_iter_cg,
                            tolerance=opt.optim.tolerance_cg)
                    else:
                        h = hessian_loaddiag_(h[None], 1e-6, 1e-8)[0]
                        distprm1 = dict(lam_dist)
                        distprm1['factor'] *= (distortion.voxel_size[contrast.readout] ** 2)
                        delta = spatial.solve_field_fmg(
                            h[None], g[None], **distprm1, dim=3,
                            bound='dft',
                            voxel_size=distortion.voxel_size,
                            nb_iter=opt.optim.max_iter_cg,
                            tolerance=opt.optim.tolerance_cg,
                            verbose=opt.verbose-1)[0]
                    delta = check_nans_(delta, warn='delta (distortion)')
                            
                    # --- line search ----------------------------------
                    armijo, armijo_prev = armijo_dist_prev[i], 0
                    ok = False
                    dd = momentum(delta)
                    dv = dd.flatten().dot(vol.flatten())
                    dd = dd.flatten().dot(dd.flatten())
                    for n_ls in range(12):
                        vol.sub_(delta, alpha=(armijo - armijo_prev))
                        armijo_prev = armijo
                        new_vreg1 = 0.5 * armijo * (armijo * dd - 2 * dv)
                        new_crit1 = derivatives_parameters(
                            contrast, distortion, intercept, maps.decay, opt,
                            do_grad=False)
                        if new_crit1 + new_vreg1 <= crit1:
                            ok = True
                            break
                        else:
                            armijo = armijo / 2
                    armijo_dist_prev[i] = armijo * 1.5
                    if not ok:
                        vol.add_(delta, alpha=armijo_prev)
                        new_crit1 = crit1
                        new_vreg1 = vreg1
                    del delta, g, h

                    new_crit += new_crit1
                    new_vreg += vreg1 + new_vreg1

                # --- track distortion improvement ---------------------
                if not ok:
                    evol = '== (x)'
                elif new_crit + new_vreg <= crit + vreg:
                    evol = f'<= ({n_ls:2d})'
                else:
                    evol = f'> ({n_ls:2d})'
                crit = new_crit
                vreg = new_vreg
                if opt.verbose:
                    ll_tmp = crit + reg + vreg + sumrls
                    pstr = (f'{n_iter_rls:3d} | {n_iter_gn:3d} | {"dist":4s} | '
                            f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} ')
                    pstr += f'+ {vreg:12.6g} '
                    pstr += f'= {ll_tmp:12.6g} | '
                    pstr += f'{evol}'
                    print(pstr)

                # --- compute GN (maps + distortion) gain --------------
                ll = crit + reg + vreg + sumrls
                ll_max = max(ll_max, ll)
                ll_prev = ll_gn[-1] if ll_gn else float('inf')
                gain = (ll_prev - ll) / (ll_max - ll_prev)
                ll_gn.append(ll)
                if opt.verbose:
                    pstr = f'{n_iter_rls:3d} | {n_iter_gn:3d} | '
                    pstr += f'{"----":4s} | '
                    pstr += f'{"-"*72:72s} | gain = {gain:7.2g}'
                    print(pstr)
                    if opt.plot:
                        _show_maps(maps, dist)
                if gain < opt.optim.tolerance_gn:
                    break

        # --------------------------------------------------------------
        #    Update RLS weights
        # --------------------------------------------------------------
        if opt.regularization.norm in ('tv', 'jtv'):
            rls_changed = True
            sumrls_prev = sumrls
            rls, sumrls = update_rls(maps, lam, opt.regularization.norm)
            sumrls = 0.5 * sumrls

            # --- compute gain -----------------------------------------
            # (we are late by one full RLS iteration when computing the
            #  gain but we save some computations)
            ll = ll_gn[-1]
            ll_prev = ll_rls[-1][-1] if ll_rls else float('inf')
            ll_rls.append(ll_gn)
            gain = (ll_prev - ll) / (ll_max - ll_prev)
            if gain < opt.optim.tolerance_rls:
                print(f'Converged ({gain:7.2g})')
                break

    # --- prepare output -----------------------------------------------
    out = postproc(maps, data)
    if opt.distortion.enable:
        out = (*out, dist)
    return out


def make_iter_rls(nb_contrasts):
    """Make it easy to iterate across RLS weights even if they are `None`."""
    def iter_rls(rls):
        if rls is None:
            for _ in range(nb_contrasts+1):
                yield None
        elif rls.dim() == 3:
            for _ in range(nb_contrasts+1):
                yield rls
        else:
            assert rls.dim() == 4
            for rls1 in rls:
                yield rls1
    return iter_rls


@torch.jit.script
def recon_fit(inter, slope, te: float):
    """Reconstruct a single echo"""
    return inter.add(slope, alpha=-te).exp()


@torch.jit.script
def ssq(x):
    """Sum of squares"""
    return (x*x).sum(dtype=torch.double)


@torch.jit.script
def dot(x, y):
    """Dot product"""
    return (x*y).sum(dtype=torch.double)


# @torch.jit.script
def get_mask_missing(dat, fit):
    """Mask of voxels excluded from the objective"""
    return ~(torch.isfinite(fit) & torch.isfinite(dat) & (dat > 0))


@torch.jit.script
def mask_nan_(x, value: float = 0.):
    """Mask out all non-finite values"""
    return x.masked_fill_(~torch.isfinite(x), value)


@torch.jit.script
def check_nans_(x, warn: Optional[str] = None, value: float = 0):
    """Mask out all non-finite values + warn if `warn is not None`"""
    msk = torch.isfinite(x)
    if warn is not None:
        if ~msk.all():
            print(f'WARNING: NaNs in {warn}')
    x.masked_fill_(~msk, value)
    return x


def derivatives_parameters(contrast, distortion, intercept, decay, opt, do_grad=True):
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
    inter = smart_pull(intercept.fdata(**backend), grid)
    slope = smart_pull(decay.fdata(**backend), grid)
    if distortion and opt.distortion.te_scaling != 'pre':
        grid_up, grid_down = distortion.exp2(
            add_identity=not opt.distortion.te_scaling)
    else:
        grid_up = grid_down = None
        
    crit = 0
    grad = torch.zeros((2,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((3,) + obs_shape, **backend) if do_grad else None

    te0 = 0
    for e, echo in enumerate(contrast):

        te = echo.te
        te0 = te0 or te
        blip = echo.blip or (2*(e % 2) - 1)
        grid_blip = grid_up if blip > 0 else grid_down
        if distortion:
            vscl = te / te0
            if opt.distortion.te_scaling == 'pre':
                vexp = distortion.iexp if blip < 0 else distortion.exp
                grid_blip = vexp(add_identity=True, alpha=vscl)
            elif opt.distortion.te_scaling == 'post':
                grid_blip = spatial.add_identity_grid_(vscl * grid_blip)

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)
        fit = recon_fit(inter, slope, te)
        pull_fit = smart_pull(fit, grid_blip, bound='dft', extrapolate=True)
        msk = get_mask_missing(dat, pull_fit)
        dat.masked_fill_(msk, 0)
        pull_fit.masked_fill_(msk, 0)
        res = dat.neg_().add_(pull_fit)
        del dat, msk, pull_fit

        # compute log-likelihood
        crit = crit + 0.5 * lam * ssq(res)

        if do_grad:
            if grid_blip is not None:
                res0 = res
                res = smart_push(res0, grid_blip, bound='dft', extrapolate=True)
                abs_res = smart_push(res0.abs_(), grid_blip, bound='dft', extrapolate=True)
                abs_res.mul_(fit)
                del res0

            # ----------------------------------------------------------
            # compute gradient and (majorised) Hessian in observed space
            #
            #   grad[inter]       =           lam * fit * res
            #   grad[decay]       =     -te * lam * fit * res
            #   hess[inter**2]    =           lam * fit * (fit + abs(res))
            #   hess[decay**2]    = (te*te) * lam * fit * (fit + abs(res))
            #   hess[inter*decay] =     -te * lam * fit * fit
            #
            # I tried to put that into an "accumulation" function but it 
            # does super weird stuff, so I keep it in the main loop. I am 
            # saving a few allocations here so I think it's faster than
            # torchscript.
            # ----------------------------------------------------------

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


def derivatives_distortion(contrast, distortion, intercept, decay, opt, do_grad=True):
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
    inter = smart_pull(intercept.fdata(**backend), grid)
    slope = smart_pull(decay.fdata(**backend), grid)
    readout = contrast.readout
    if opt.distortion.te_scaling != 'pre':
        grid_up, grid_down = distortion.exp2(
            add_identity=not opt.distortion.te_scaling)
    else:
        grid_up = grid_down = None

    crit = 0
    grad = torch.zeros(obs_shape + (3,), **backend) if do_grad else None
    hess = torch.zeros(obs_shape + (6,), **backend) if do_grad else None

    te0 = 0
    for e, echo in enumerate(contrast):

        te = echo.te
        te0 = te0 or te
        blip = echo.blip or (2*(e % 2) - 1)
        grid_blip = grid_up if blip > 0 else grid_down
        vscl = te / te0
        if opt.distortion.te_scaling == 'pre':
            vexp = distortion.iexp if blip < 0 else distortion.exp
            grid_blip = vexp(add_identity=True, alpha=vscl)
        elif opt.distortion.te_scaling:
            grid_blip = spatial.add_identity_grid_(vscl * grid_blip)

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)  # observed
        fit = recon_fit(inter, slope, te)                    # fitted
        if do_grad and isinstance(distortion, DenseDeformation):
            # D(fit) o phi
            if not (grid_blip == 0).any():
                gfit = spatial.diff(fit, bound='dft', dim=[-3, -2, -1])
            else:
                gfit = smart_grad(fit, grid_blip, bound='dft', extrapolate=True)
        fit = smart_pull(fit, grid_blip, bound='dft', extrapolate=True)
        if do_grad and isinstance(distortion, SVFDeformation):
            # D(fit o phi)
            gfit = spatial.diff(fit, bound='dft', dim=[-3, -2, -1])
        msk = get_mask_missing(dat, fit)    # mask of missing values
        dat.masked_fill_(msk, 0)
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
                if opt.distortion.te_scaling == 'pre':
                    vel = ((-vscl) * vel) if blip < 0 else (vscl * vel)
                elif blip < 0:
                    vel = -vel
                g1, h1 = spatial.exp_backward(vel, g1, h1, steps=distortion.steps)
            if blip < 0:
                g1 = g1.neg_()

            alpha_g = alpha_h = lam
            if opt.distortion.te_scaling == 'pre':
                alpha_g = alpha_g * vscl
                alpha_h = alpha_h * (vscl * vscl)
            grad.add_(g1, alpha=alpha_g)
            hess.add_(h1, alpha=alpha_h)

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


def solve_parameters(hess, grad, rls, lam, vx, opt):
    """Solve the regularized linear system

    Parameters
    ----------
    hess : (2*P+1, *shape) tensor
    grad : (P+1, *shape) tensor
    rls : ([P+1], *shape) tensor or None
    lam : (P,) sequence[float]
    vx : (D,) sequence[float]
    opt : Options

    Returns
    -------
    delta : (P+1, *shape) tensor

    """
    # The ESTATICS Hessian has a very particular form (intercepts do not
    # have cross elements). We therefore need to tell the solver how to operate
    # on it.

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

    return spatial.solve_field_fmg(hess, grad, rls, factor=lam, membrane=1,
                                   voxel_size=vx, verbose=opt.verbose - 1,
                                   nb_iter=opt.optim.max_iter_cg,
                                   tolerance=opt.optim.tolerance_cg,
                                   matvec=matvec, matsolve=matsolve, matdiag=matdiag)

    # # ----------------------------------------------------------------
    # # In the 'Model-based MPM` paper, we used to solve the linear
    # # system using PCG. I know use a multi-grid version of it, which
    # # I found to yield lower residual errors.
    # # Uncomment this line to use vanilla PCG instead.
    # # ----------------------------------------------------------------
    # return spatial.solve_field(hess, grad, rls, dim=3, factor=lam, membrane=1,
    #                            voxel_size=vx, verbose=opt.verbose - 1,
    #                            max_iter=32, tolerance=1e-5, stop='a',
    #                            matvec=matvec, matsolve=matsolve, matdiag=matdiag)

    # # ----------------------------------------------------------------
    # # This is the old version of the code, before it got refactored
    # # and generalized in the `spatial` module.
    # # ----------------------------------------------------------------

    # def hess_fn(x):
    #     result = hessian_matmul(hess, x)
    #     if not opt.regularization.norm:
    #         return result
    #     for i, (map, weight, l) in enumerate(zip(x, rls, lam)):
    #         if not l:
    #             continue
    #         _, res1 = _nonlin_reg(map, vx, weight, l)
    #         result[i] += res1
    #     return result
    #
    # # The Hessian is A = H + L, where H corresponds to the data term
    # # and L to the regularizer. Note that, L = D'WD where D is the
    # # gradient operator, D' the divergence and W a diagonal matrix
    # # that contains the RLS weights.
    # # We use (H + diag(|D'D|w)) as a preconditioner because it is easy to
    # # invert and majorises the true Hessian.
    # hessp = hess.clone()
    # smo = torch.as_tensor(vx).square().reciprocal().sum().item()
    # for i, (weight, l) in enumerate(zip(rls, lam)):
    #     if not l:
    #         continue
    #     hessp[2*i] += l * (rls_maj(weight, vx) if weight is not None else 4*smo)
    #
    # def precond(x):
    #     return hessian_solve(hessp, x)
    #
    # result = core.optim.cg(hess_fn, grad, precond=precond,
    #                        verbose=(opt.verbose > 1), stop='norm',
    #                        max_iter=32, #opt.optim.max_iter_cg,
    #                        tolerance=1e-5) #opt.optim.tolerance_cg)
    # return result


def update_rls(maps, lam=1., norm='jtv'):
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
        (Inverted) Weights from the reweighted least squares scheme
    sumrls : () tensor
        Sum of the (non-inverted) weights
    """
    # vx = spatial.voxel_size(maps.affine)
    # return spatial.membrane_weights(maps.volume, dim=3, factor=lam,
    #                                 joint=(norm == 'jtv'), voxel_size=vx,
    #                                 return_sum=True)

    # ----------------------------------------------------------------
    # This is the old version of the code, before it got refactored
    # and generalized in the `spatial` module.
    # ----------------------------------------------------------------

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
            rls1 = update_rls(map, l, '__internal__')
            rls1 = rls1.sqrt_()
            rls.append(rls1)
    else:
        assert norm == 'jtv'
        rls = 0
        for map, l in zip(maps, lam):
            rls += update_rls(map, l, '__internal__')
        rls = rls.sqrt_()

    sumrls = rls.sum(dtype=torch.double)
    eps = core.constants.eps(rls.dtype)
    rls = rls.clamp_min_(eps).reciprocal_()

    return rls, sumrls


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
