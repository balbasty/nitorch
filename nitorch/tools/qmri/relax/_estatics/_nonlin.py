import torch
from nitorch import core, spatial
from ._options import ESTATICSOptions
from ._preproc import preproc, postproc
from ._utils import (hessian_loaddiag_, hessian_matmul, hessian_solve)
from ..utils import smart_pull, smart_push, smart_grad, smart_grid
from nitorch.core.math import besseli, besseli_ratio
from nitorch.tools.qmri.param import ParameterMap, SVFDeformation, DenseDeformation
from typing import Optional

# Boundary condition used for the distortion field throughout
DIST_BOUND = 'dct2'


def _get_level(level, aff0, shape0):
    """Get shape and affine of a given resolution level"""
    return spatial.affine_resize(aff0, shape0, 1/(2 ** (level-1)))


def _resize(maps, rls, aff, shape):
    """Resize (prolong) current maps to a target resolution"""
    for map in maps:
        map.volume = spatial.resize(map.volume[None, None, ...],
                                    shape=shape)[0, 0]
        map.affine = aff
    maps.affine = aff
    if rls is not None:
        if rls.dim() == len(shape):
            rls = spatial.resize(rls[None, None], shape=shape)[0, 0]
        else:
            rls = spatial.resize(rls[None], shape=shape)[0]
    return maps, rls


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
    if len(data) > 1:
        pstr = f'Fitting a (shared) exponential decay model with {len(data)} contrasts.'
    else:
        pstr = f'Fitting an exponential decay model.'
    print(pstr)
    print('Echo times:')
    for i, contrast in enumerate(data):
        print(f'    - contrast {i:2d}: [' + ', '.join([f'{te*1e3:.1f}' for te in contrast.te]) + '] ms')

    # --- estimate noise / register / initialize maps ------------------
    data, maps, dist = preproc(data, opt)
    vx = spatial.voxel_size(maps.affine)
    nb_contrasts = len(maps) - 1

    if opt.distortion.enable:
        print('Readout directions:')
        for i, contrast in enumerate(data):
            layout = spatial.affine_to_layout(contrast.affine)
            layout = spatial.volume_layout_to_name(layout)
            readout = layout[contrast.readout]
            readout = ('left-right' if 'L' in readout or 'R' in readout else
                       'infra-supra' if 'I' in readout or 'S' in readout else
                       'antero-posterior' if 'A' in readout or 'P' in readout else
                       'unknown')
            print(f'    - contrast {i:2d}: {readout}')

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

    else:
        print('Without distortion correction')

    # --- initialize nb of iterations ----------------------------------
    if not opt.regularization.norm.endswith('tv'):
        # no reweighting -> do more gauss-newton updates instead
        opt.optim.max_iter_gn *= opt.optim.max_iter_rls
        opt.optim.max_iter_rls = 1
    if not opt.distortion.enable:
        # no distortion -> merge inner and outer loops
        opt.optim.max_iter_gn *= opt.optim.max_iter_prm
        opt.optim.max_iter_prm = 1
    print('Optimization:')
    print(f'    - Tolerance:        {opt.optim.tolerance}')
    if opt.regularization.norm.endswith('tv'):
        print(f'    - IRLS iterations:  {opt.optim.max_iter_rls}')
    print(f'    - GN iterations:    {opt.optim.max_iter_gn}')
    if opt.distortion.enable:
        print(f'    - Param iterations: {opt.optim.max_iter_prm}')
        print(f'    - Dist iterations:  {opt.optim.max_iter_dist}')
    print(f'    - FMG cycles:       2')
    print(f'    - CG iterations:    {opt.optim.max_iter_cg}'
          f' (tolerance: {opt.optim.tolerance_cg})')
    if opt.optim.nb_levels > 1:
        print(f'    - Levels:           {opt.optim.nb_levels}')

    # ------------------------------------------------------------------
    #                     MAIN OPTIMIZATION LOOP
    # ------------------------------------------------------------------

    if opt.verbose:
        pstr = f'{"rls":^3s} | {"gn":^3s} | {"step":^4s} | '
        pstr += f'{"fit":^12s} + {"reg":^12s} + {"rls":^12s} '
        if opt.distortion.enable:
            pstr += f'+ {"dist":^12s} '
        pstr += f'= {"crit":^12s}'
        print('\n' + pstr)
        print('-' * len(pstr))

    # --- initialize tracking of the objective function ----------------
    crit = float('inf')
    vreg = 0
    reg = 0
    sumrls_prev = sumrls
    rls_changed = False
    armijo_dist_prev = [1] * nb_contrasts
    ll_gn = ll_rls = float('inf')
    ll_scl = sum(core.py.prod(dat.shape) for dat in data)

    # --- Multi-Resolution loop ----------------------------------------
    shape0 = shape = maps.shape[1:]
    aff0 = aff = maps.affine
    vx0 = vx = spatial.voxel_size(aff0)
    scl0 = vx0.prod()
    scl = vx.prod() / scl0
    for level in range(opt.optim.nb_levels, 0, -1):

        if opt.optim.nb_levels > 1:
            aff, shape = _get_level(level, aff0, shape0)
            vx = spatial.voxel_size(aff)
            scl = vx.prod() / scl0
            maps, rls = _resize(maps, rls, aff, shape)
            if opt.regularization.norm in ('tv', 'jtv'):
                sumrls = 0.5 * scl * rls.reciprocal().sum(dtype=torch.double)

        grad = torch.empty((len(data) + 1, *shape), **backend)
        hess = torch.empty((len(data) * 2 + 1, *shape), **backend)

        # --- RLS loop -----------------------------------------------------
        #   > max_iter_rls == 1 if regularization is not (J)TV
        max_iter_rls = opt.optim.max_iter_rls
        if level != 1:
            max_iter_rls = 1
        for n_iter_rls in range(1, max_iter_rls+1):

            # --- Gauss-Newton ---------------------------------------------
            # . If distortion correction is enabled, we perform one step of
            #   parameter update and one step of distortion update per iteration.
            # . We keep track of the objective within each RLS iteration so
            #   that we can stop Gauss-Newton early if we it reaches a plateau.
            max_iter_gn = opt.optim.max_iter_gn
            for n_iter_gn in range(1, max_iter_gn+1):

                # ----------------------------------------------------------
                #    Update parameter maps
                # ----------------------------------------------------------

                max_iter_prm = opt.optim.max_iter_prm
                if n_iter_gn == 1 and n_iter_rls == 1:
                    max_iter_prm = max_iter_prm * 2
                for n_iter_prm in range(1, max_iter_prm + 1):

                    crit_prev = crit
                    reg_prev = reg

                    # --- loop over contrasts ------------------------------
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
                    del g1, h1

                    # --- regularization -----------------------------------
                    reg = 0.
                    if opt.regularization.norm:
                        for i, (map, weight, l) \
                                in enumerate(zip(maps, iter_rls(rls), lam)):
                            if not l:
                                continue
                            g1 = spatial.membrane(map.volume, weights=weight, dim=3,
                                                  voxel_size=vx).mul_(l*scl)
                            reg1 = 0.5 * dot(map.volume, g1)
                            reg += reg1
                            grad[i] += g1
                        del g1

                    # --- track RLS improvement ----------------------------
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

                    # --- gauss-newton -------------------------------------
                    # Computing the GN step involves solving H\g
                    hess = check_nans_(hess, warn='hessian')
                    hess = hessian_loaddiag_(hess, 1e-6, 1e-8)
                    lam_ = [l*scl for l in lam]
                    deltas = solve_parameters(hess, grad, rls, lam_, vx, opt)
                    deltas = check_nans_(deltas, warn='delta')

                    if not opt.distortion.enable:
                        # No need for a line search
                        for map, delta in zip(maps, deltas):
                            map.volume -= delta
                            if map.min is not None or map.max is not None:
                                map.volume.clamp_(map.min, map.max)
                        del deltas

                    else:  # === distortion.enabled ========================

                        # --- line search ----------------------------------
                        # Block coordinate descent makes it a bit less robust
                        # -> we use a line search
                        dd = []
                        if opt.regularization.norm:
                            for i, (delta, weight, l) \
                                    in enumerate(zip(deltas, iter_rls(rls), lam)):
                                if not l:
                                    continue
                                g1 = spatial.membrane(delta, weights=weight,
                                                      dim=3, voxel_size=vx).mul_(l*scl)
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
                            new_reg = 0
                        
                        for map in maps:
                            if map.min is not None or map.max is not None:
                                map.volume.clamp_(map.min, map.max)

                        new_reg = reg + new_reg

                        # --- track parameter map improvement --------------
                        if not ok:
                            evol = '== (x)'
                        elif new_crit + new_reg <= crit + reg:
                            evol = f'<= ({n_ls:2d})'
                        else:
                            evol = f'> ({n_ls:2d})'
                        gain = (crit + reg) - (new_crit + new_reg)
                        crit = new_crit
                        reg = new_reg
                        if opt.verbose:
                            ll_tmp = crit + reg + vreg + sumrls
                            pstr = (f'{n_iter_rls:3d} | {n_iter_gn:3d} | {"prm":4s} | '
                                    f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} ')
                            if opt.distortion.enable:
                                pstr += f'+ {vreg:12.6g} '
                            pstr += f'= {ll_tmp:12.6g} | gain = {gain/ll_scl:7.2g} | '
                            pstr += f'{evol}'
                            print(pstr)
                            if opt.plot:
                                _show_maps(maps, dist, data)
                        if not ok:
                            break
                        if n_iter_prm > 1 and gain < opt.optim.tolerance * ll_scl:
                            break

                # ----------------------------------------------------------
                #    Update distortions
                # ----------------------------------------------------------
                max_iter_dist = opt.optim.max_iter_dist
                if not opt.distortion.enable:
                    max_iter_dist = 0
                for n_iter_dist in range(1, max_iter_dist + 1):

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
                                    vol, **lam_dist, bound=DIST_BOUND,
                                    voxel_size=distortion.voxel_size)
                            else:
                                distprm1 = dict(lam_dist)
                                distprm1['factor'] *= distortion.voxel_size[contrast.readout] ** 2
                                g1 = spatial.regulariser(
                                    vol[None], **distprm1, dim=3, bound=DIST_BOUND,
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
                            h = hessian_loaddiag_(h, 1e-32, 1e-32)
                            h = core.utils.movedim(h, -4, -1)
                            delta = spatial.solve_grid_fmg(
                                h, g, **lam_dist, bound=DIST_BOUND,
                                voxel_size=distortion.voxel_size,
                                verbose=max(0, opt.verbose-1),
                                nb_iter=opt.optim.max_iter_cg,
                                tolerance=opt.optim.tolerance_cg)
                        else:
                            h = hessian_loaddiag_(h[None], 1e-32, 1e-32)[0]
                            distprm1 = dict(lam_dist)
                            distprm1['factor'] *= (distortion.voxel_size[contrast.readout] ** 2)
                            delta = spatial.solve_field_fmg(
                                h[None], g[None], **distprm1, dim=3,
                                bound=DIST_BOUND,
                                voxel_size=distortion.voxel_size,
                                nb_iter=opt.optim.max_iter_cg,
                                tolerance=opt.optim.tolerance_cg,
                                verbose=max(0, opt.verbose-1))[0]
                        delta = check_nans_(delta, warn='delta (distortion)')

                        # --- line search ----------------------------------
                        armijo, armijo_prev = 1, 0 # armijo_dist_prev[i], 0
                        ok = False
                        dd = momentum(delta)
                        dv = dd.flatten().dot(vol.flatten())
                        dd = dd.flatten().dot(delta.flatten())
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
                        armijo_dist_prev[i] = armijo * 2
                        if not ok:
                            vol.add_(delta, alpha=armijo_prev)
                            new_crit1 = crit1
                            new_vreg1 = 0
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
                    gain = (crit + vreg) - (new_crit + new_vreg)
                    crit = new_crit
                    vreg = new_vreg
                    if opt.verbose:
                        ll_tmp = crit + reg + vreg + sumrls
                        pstr = (f'{n_iter_rls:3d} | {n_iter_gn:3d} | {"dist":4s} | '
                                f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} ')
                        pstr += f'+ {vreg:12.6g} '
                        pstr += f'= {ll_tmp:12.6g} | gain = {gain/ll_scl:7.2g} | '
                        pstr += f'{evol}'
                        print(pstr)
                        if opt.plot:
                            _show_maps(maps, dist, data)
                    if not ok:
                        break

                    if n_iter_dist > 1 and gain < opt.optim.tolerance * ll_scl:
                        break

                # --- compute GN (maps + distortion) gain --------------
                ll = crit + reg + vreg + sumrls
                ll_gn_prev = ll_gn
                ll_gn = ll
                gain = ll_gn_prev - ll_gn
                if opt.verbose:
                    pstr = f'{n_iter_rls:3d} | {n_iter_gn:3d} | '
                    if opt.distortion.enable:
                        pstr += f'{"----":4s} | '
                        pstr += f'{"-"*72:72s} | '
                    else:
                        pstr += f'{"prm":4s} | '
                        pstr += f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} '
                        pstr += f'= {ll:12.6g} | '
                    pstr += f'gain = {gain/ll_scl:7.2g}'
                    print(pstr)
                    if opt.plot:
                        _show_maps(maps, dist, data)
                if gain < opt.optim.tolerance * ll_scl:
                    break

            # --------------------------------------------------------------
            #    Update RLS weights
            # --------------------------------------------------------------
            if level == 1 and opt.regularization.norm in ('tv', 'jtv'):
                rls_changed = True
                sumrls_prev = sumrls
                rls, sumrls = update_rls(maps, lam, opt.regularization.norm)
                sumrls = 0.5 * sumrls

                # --- compute gain -----------------------------------------
                # (we are late by one full RLS iteration when computing the
                #  gain but we save some computations)
                ll_rls_prev = ll_rls
                ll_rls = ll_gn
                gain = ll_rls_prev - ll_rls
                if gain < opt.optim.tolerance * ll_scl:
                    print(f'Converged ({gain/ll_scl:7.2g})')
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


def get_mask_missing(dat, fit):
    """Mask of voxels excluded from the objective"""
    return ~(torch.isfinite(fit) & torch.isfinite(dat) & (dat > 0))


def mask_nan_(x, value: float = 0.):
    """Mask out all non-finite values"""
    return x.masked_fill_(torch.isfinite(x).bitwise_not(), value)


def check_nans_(x, warn: Optional[str] = None, value: float = 0):
    """Mask out all non-finite values + warn if `warn is not None`"""
    msk = torch.isfinite(x)
    if warn is not None:
        if ~(msk.all()):
            print(f'WARNING: NaNs in {warn}')
    x.masked_fill_(msk.bitwise_not(), value)
    return x


def nll_chi(dat, fit, msk, lam, df, return_residuals=True):
    """Negative log-likelihood of the noncentral Chi distribution

    Parameters
    ----------
    dat : tensor
        Observed data -- will be modified in-place
    fit : tensor
        Signal fit
    msk : tensor
        Mask of observed values
    lam : float
        Noise precision
    df : float
        Degrees of freedom
    return_residuals : bool
        Return residuals (gradient) on top of nll

    Returns
    -------
    nll : () tensor
        Negative log-likelihood
    res : tensor, if `return_residuals`
        Residuals

    """
    fitm = fit[msk]
    datm = dat[msk]

    # components of the log-likelihood
    sumlogfit = fitm.clamp_min(1e-32).log_().sum(dtype=torch.double)
    sumfit2 = fitm.flatten().dot(fitm.flatten())
    sumlogdat = datm.clamp_min(1e-32).log_().sum(dtype=torch.double)
    sumdat2 = datm.flatten().dot(datm.flatten())

    # reweighting
    z = (fitm * datm).mul_(lam).clamp_min_(1e-32)
    xi = besseli_ratio(df / 2 - 1, z)
    logbes = besseli(df / 2 - 1, z, 'log')
    logbes = logbes.sum(dtype=torch.double)

    # sum parts
    crit = (df / 2 - 1) * sumlogfit - (df / 2) * sumlogdat - logbes
    crit += 0.5 * lam * (sumfit2 + sumdat2)
    if not return_residuals:
        return crit

    # compute residuals
    res = dat.zero_()
    res[msk] = datm.mul_(xi).neg_().add_(fitm)
    return crit, res


def nll_gauss(dat, fit, msk, lam, return_residuals=True):
    """Negative log-likelihood of the noncentral Chi distribution

    Parameters
    ----------
    dat : tensor
        Observed data (should be zero where not observed)
    fit : tensor
        Signal fit (should be zero where not observed)
    msk : tensor
        Mask of observed values
    lam : float
        Noise precision
    nu : float
        Degrees of freedom
    return_residuals : bool
        Return residuals (gradient) on top of nll

    Returns
    -------
    nll : () tensor
        Negative log-likelihood
    res : tensor, if `return_residuals`
        Residuals

    """
    res = dat.neg_().add_(fit)
    crit = 0.5 * lam * ssq(res[msk])
    return (crit, res) if return_residuals else crit


# if core.utils.torch_version('>', (1, 4)):
    # For some reason, the output of torch.isfinite is not understood
    # as a tensor by TS. I am disabling TS for these functions until
    # I find a better solution.
    # get_mask_missing = torch.jit.script(get_mask_missing)
    # mask_nan_ = torch.jit.script(mask_nan_)
    # check_nans_ = torch.jit.script(check_nans_)


def derivatives_parameters(contrast, distortion, intercept, decay, opt,
                           do_grad=True):
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
    lam = 1 / contrast.noise
    df = contrast.dof
    chi = opt.likelihood[0].lower() == 'c'
    readout = contrast.readout

    # pull parameter maps to observed space
    grid = smart_grid(aff, obs_shape, recon_shape)
    inter = smart_pull(intercept.fdata(**backend), grid)
    slope = smart_pull(decay.fdata(**backend), grid)
    if distortion:
        grid_up, grid_down, jac_up, jac_down = distortion.exp2(
            add_identity=True, jacobian=True)
        jac_up = jac_up[..., readout, readout]
        jac_down = jac_down[..., readout, readout]
    else:
        grid_up = grid_down = jac_up = jac_down = None
        
    crit = 0
    grad = torch.zeros((2,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((3,) + obs_shape, **backend) if do_grad else None

    te0 = 0
    for e, echo in enumerate(contrast):

        te = echo.te
        te0 = te0 or te
        blip = echo.blip or (2*(e % 2) - 1)
        grid_blip = grid_up if blip > 0 else grid_down
        jac_blip = jac_up if blip > 0 else jac_down

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)
        fit = recon_fit(inter, slope, te)
        # push_fit = smart_push(fit, grid_blip, bound='dft', extrapolate=True)
        push_fit = smart_pull(fit, grid_blip, bound='dft', extrapolate=True)
        if jac_blip is not None:
            push_fit = push_fit * jac_blip
        msk = get_mask_missing(dat, push_fit)
        dat.masked_fill_(msk, 0)
        push_fit.masked_fill_(msk, 0)
        msk = msk.bitwise_not_()

        if chi:
            crit1, res = nll_chi(dat, push_fit, msk, lam, df)
        else:
            crit1, res = nll_gauss(dat, push_fit, msk, lam)
        del dat, push_fit
        crit += crit1

        if do_grad:
            msk = msk.to(fit.dtype)
            if grid_blip is not None:
                res0 = res
                # res = smart_pull(res0, grid_blip, bound='dft', extrapolate=True)
                res = smart_push(res0*jac_blip, grid_blip, bound='dft', extrapolate=True)
                # abs_res = smart_pull(res0.abs_(), grid_blip, bound='dft', extrapolate=True)
                abs_res = smart_push(res0.abs_().mul_(jac_blip), grid_blip, bound='dft', extrapolate=True)
                abs_res.mul_(fit)
                # msk = smart_pull(msk, grid_blip, bound='dft', extrapolate=True)
                msk = msk.mul_(jac_blip).mul_(jac_blip)
                msk = smart_push(msk, grid_blip, bound='dft', extrapolate=True)
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
            fit2 = fit.mul_(fit).mul_(msk)
            del msk
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


def derivatives_distortion(contrast, distortion, intercept, decay, opt,
                           do_grad=True):
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
    df = contrast.dof
    chi = opt.likelihood[0].lower() == 'c'
    readout = contrast.readout

    # pull parameter maps to observed space
    grid = smart_grid(aff, obs_shape, recon_shape)
    inter = smart_pull(intercept.fdata(**backend), grid)
    slope = smart_pull(decay.fdata(**backend), grid)
    grid_up, grid_down, jac_up, jac_down = distortion.exp2(
        add_identity=True, jacobian=True)
    jac_up = jac_up[..., readout, readout]
    jac_down = jac_down[..., readout, readout]

    crit = 0
    grad = torch.zeros(obs_shape + (3,), **backend) if do_grad else None
    hess = torch.zeros(obs_shape + (6,), **backend) if do_grad else None

    te0 = 0
    for e, echo in enumerate(contrast):

        te = echo.te
        te0 = te0 or te
        blip = echo.blip or (2*(e % 2) - 1)
        grid_blip = grid_up if blip > 0 else grid_down
        jac_blip = jac_up if blip > 0 else jac_down

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)  # observed
        fit = recon_fit(inter, slope, te)                    # fitted
        if do_grad and isinstance(distortion, DenseDeformation):
            # D(fit) o phi
            # gfit = smart_grad(fit, igrid_blip, bound='dft', extrapolate=True).neg_()
            gfit = smart_grad(fit, grid_blip, bound='dft', extrapolate=True)
            gfit *= jac_blip.unsqueeze(-1)
        # fit = smart_push(fit, grid_blip, bound='dft', extrapolate=True)
        fit = smart_pull(fit, grid_blip, bound='dft', extrapolate=True)
        fit.mul_(jac_blip)
        msk = get_mask_missing(dat, fit)    # mask of missing values
        if do_grad and isinstance(distortion, SVFDeformation):
            # D(fit o phi)
            gfit = spatial.diff(fit, bound='dft', dim=[-3, -2, -1])
            gfit.masked_fill_(msk.unsqueeze(-1), 0)
        dat.masked_fill_(msk, 0)
        fit.masked_fill_(msk, 0)
        msk = msk.bitwise_not_()

        if chi:
            crit1, res = nll_chi(dat, fit, msk, lam, df)
        else:
            crit1, res = nll_gauss(dat, fit, msk, lam)
        del dat, fit, msk
        crit += crit1

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

            alpha_g = alpha_h = lam
            alpha_g = alpha_g * blip
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
                                   voxel_size=vx, verbose=max(0, opt.verbose - 1),
                                   nb_iter=opt.optim.max_iter_cg,
                                   tolerance=opt.optim.tolerance_cg,
                                   matvec=matvec, matsolve=matsolve, matdiag=matdiag)

    # # ----------------------------------------------------------------
    # # In the 'Model-based MPM` paper, we used to solve the linear
    # # system using PCG. I now use a multi-grid version of it, which
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


def _show_maps(maps, dist, data):
    import matplotlib.pyplot as plt
    from nitorch.plot.colormaps import disp_to_rgb
    has_dist = any([d is not None for d in dist])
    ncol = max(len(maps), len(dist))
    for i, map in enumerate(maps):
        plt.subplot(1 + has_dist, ncol, i+1)
        vol = map.volume[:, :, map.shape[-1]//2]
        if i < len(maps) - 1:
            vol = vol.exp()
        plt.imshow(vol.cpu())
        plt.axis('off')
        plt.colorbar()
    if has_dist:
        for i, (dat, dst) in enumerate(zip(data, dist)):
            if dst is None:
                continue
            readout = dat.readout
            vol = dst.volume
            plt.subplot(1 + has_dist, ncol, i+1+ncol)
            vol = vol[:, :, dst.shape[-2]//2]
            if readout is not None:
                vol = vol[..., readout]
            else:
                vol = vol.square().sum(-1).sqrt()
            plt.imshow(vol.cpu())
            plt.axis('off')
            plt.colorbar()
    plt.show()

    
def _show_grad_dist(grad, hess, delta):
    import matplotlib.pyplot as plt

    grad = grad[:, :, grad.shape[-1]//2]
    hess = hess[:, :, hess.shape[-1]//2]
    delta = delta[:, :, delta.shape[-1]//2]
    delta0 = grad/hess
    delta0[hess < 1e-6] = 0

    plt.subplot(1, 4, 1)
    plt.imshow(grad.cpu())
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 4, 2)
    plt.imshow(hess.cpu())
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 4, 3)
    plt.imshow(delta0.cpu())
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 4, 4)
    plt.imshow(delta.cpu())
    plt.axis('off')
    plt.colorbar()
    plt.show()