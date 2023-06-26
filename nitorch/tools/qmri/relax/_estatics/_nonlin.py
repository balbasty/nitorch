import torch
from nitorch import core, spatial
from ._options import ESTATICSOptions
from ._preproc import preproc, postproc
from ._utils import (hessian_loaddiag_)
from ..utils import (smart_pull, smart_push, smart_grid, pull1d, push1d,
                     nll_chi, nll_gauss, dot,
                     get_mask_missing, mask_nan_, check_nans_)
from nitorch.tools.qmri.param import ParameterMap, SVFDistortion, DenseDistortion

# Boundary condition used for the distortion field throughout
DIST_BOUND = 'dct2'


def div1d(disp, dim):
    return spatial.div1d(disp, dim=dim, bound=DIST_BOUND, side='c')


def _get_level(level, aff0, shape0):
    """Get shape and affine of a given resolution level"""
    return spatial.affine_resize(aff0, shape0, 1/(2 ** (level-1)))


def _resize(maps, rls, aff, shape):
    """Resize (prolong) current maps to a target resolution"""
    maps.volume = spatial.resize(maps.volume[None], shape=shape)[0]
    maps.affine = aff
    # for map in maps:
    #     map.volume = spatial.resize(map.volume[None, None, ...],
    #                                 shape=shape)[0, 0]
    #     map.affine = aff
    # maps.affine = aff
    if rls is not None:
        if rls.dim() == len(shape):
            rls = spatial.resize(rls[None, None], shape=shape)[0, 0]
        else:
            rls = spatial.resize(rls[None], shape=shape)[0]
    return maps, rls


def _prepare(data, dist, opt):

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
    data, maps, dist = preproc(data, dist, opt)
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
    opt.regularization.factor = lam
    # 2. Distortion fields regularization
    lam_dist = dict(
        factor=opt.distortion.factor,
        absolute=opt.distortion.absolute,
        membrane=opt.distortion.membrane,
        bending=opt.distortion.bending)
    opt.distortion.factor = lam_dist

    # --- initialize weights (RLS) -------------------------------------
    mean_shape = maps.decay.volume.shape
    rls = None
    if opt.regularization.norm.endswith('tv'):
        rls_shape = mean_shape
        if opt.regularization.norm == 'tv':
            rls_shape = (len(maps), *rls_shape)
        rls = ParameterMap(rls_shape, fill=1, **backend).volume

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
        print(f'    - absolute:       {opt.distortion.absolute * opt.distortion.factor["factor"]}')
        print(f'    - membrane:       {opt.distortion.membrane * opt.distortion.factor["factor"]}')
        print(f'    - bending:        {opt.distortion.bending * opt.distortion.factor["factor"]}')

    else:
        print('Without distortion correction')

    # --- initialize nb of iterations ----------------------------------
    if not opt.regularization.norm.endswith('tv'):
        # no reweighting -> do more gauss-newton updates instead
        opt.optim.max_iter_prm *= opt.optim.max_iter_rls
        opt.optim.max_iter_rls = 1
    print('Optimization:')
    print(f'    - Tolerance:        {opt.optim.tolerance}')
    if opt.regularization.norm.endswith('tv'):
        print(f'    - IRLS iterations:  {opt.optim.max_iter_rls}')
    print(f'    - Param iterations: {opt.optim.max_iter_prm}')
    if opt.distortion.enable:
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
        if opt.optim.nb_levels > 1:
            pstr = f'{"lvl":3s} | ' + pstr
        print('\n' + pstr)
        print('-' * len(pstr))

    return data, maps, dist, opt, rls


def nonlin(data, dist=None, opt=None):
    """Fit the ESTATICS model to multi-echo Gradient-Echo data.

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
        Observed GRE data.
    dist : sequence[Optional[ParameterizedDistortion]], optional
        Pre-computed distortion fields
    opt : Options, optional
        Algorithm options.

    Returns
    -------
    intecepts : sequence[GradientEcho]
        Echo series extrapolated to TE=0
    decay : estatics.ParameterMap
        R2* decay map
    distortions : sequence[ParameterizedDistortion], if opt.distortion.enable
        B0-induced distortion fields

    """
    if opt.distortion.enable:
        return meetup(data, dist, opt)

    # --- prepare everything -------------------------------------------
    data, maps, dist, opt, rls = _prepare(data, dist, opt)
    sumrls = 0.5 * rls.sum(dtype=torch.double) if rls is not None else 0
    backend = dict(dtype=opt.backend.dtype, device=opt.backend.device)

    # --- initialize tracking of the objective function ----------------
    crit = float('inf')
    vreg = 0
    reg = 0
    sumrls_prev = sumrls
    rls_changed = False
    ll_gn = ll_rls = float('inf')
    ll_scl = sum(core.py.prod(dat.shape) for dat in data)

    # --- Multi-Resolution loop ----------------------------------------
    shape0 = shape = maps.shape[1:]
    aff0 = aff = maps.affine
    lam0 = lam = opt.regularization.factor
    vx0 = vx = spatial.voxel_size(aff0)
    scl0 = vx0.prod()
    for level in range(opt.optim.nb_levels, 0, -1):

        if opt.optim.nb_levels > 1:
            aff, shape = _get_level(level, aff0, shape0)
            vx = spatial.voxel_size(aff)
            scl = vx.prod() / scl0
            lam = [float(l*scl) for l in lam0]
            maps, rls = _resize(maps, rls, aff, shape)
            if opt.regularization.norm in ('tv', 'jtv'):
                sumrls = 0.5 * scl * rls.reciprocal().sum(dtype=torch.double)

        grad = torch.empty((len(data) + 1, *shape), **backend)
        hess = torch.empty((len(data) * 2 + 1, *shape), **backend)

        # --- RLS loop -------------------------------------------------
        #   > max_iter_rls == 1 if regularization is not (J)TV
        max_iter_rls = opt.optim.max_iter_rls
        if level != 1:
            max_iter_rls = 1
        for n_iter_rls in range(1, max_iter_rls+1):

            # --- helpers ----------------------------------------------
            regularizer = lambda x: spatial.regulariser(
                x, weights=rls, dim=3, voxel_size=vx, membrane=1, factor=lam)
            solve = lambda h, g: spatial.solve_field(
                h, g, rls, factor=lam,
                membrane=1 if opt.regularization.norm else 0,
                voxel_size=vx, max_iter=opt.optim.max_iter_cg,
                tolerance=opt.optim.tolerance_cg, dim=3)
            reweight = lambda x, **k: spatial.membrane_weights(
                x, lam, vx, dim=3, **k,
                joint=opt.regularization.norm == 'jtv',
                eps=core.constants.eps(rls.dtype))

            # ----------------------------------------------------------
            #    Update parameter maps
            # ----------------------------------------------------------

            max_iter_prm = opt.optim.max_iter_prm
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
                    grad[gind] += g1
                    hind = [i, len(grad)-1, len(grad)+i]
                    hess[hind] += h1
                    crit += crit1
                del g1, h1

                # --- regularization -----------------------------------
                reg = 0.
                if opt.regularization.norm:
                    g1 = regularizer(maps.volume)
                    reg = 0.5 * dot(maps.volume, g1)
                    grad += g1
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
                    pstr = (f'{n_iter_rls:3d} | {"---":3s} | {"rls":4s} | '
                            f'{crit_prev:12.6g} + {reg:12.6g} + {sumrls:12.6g} ')
                    if opt.distortion.enable:
                        pstr += f'+ {vreg:12.6g} '
                    pstr += f'= {ll_tmp:12.6g} | '
                    pstr += f'{evol}'
                    if opt.optim.nb_levels > 1:
                        pstr = f'{level:3d} | ' + pstr
                    print(pstr)

                # --- gauss-newton -------------------------------------
                # Computing the GN step involves solving H\g
                hess = check_nans_(hess, warn='hessian')
                hess = hessian_loaddiag_(hess, 1e-6, 1e-8)
                deltas = solve(hess, grad)
                deltas = check_nans_(deltas, warn='delta')

                for map, delta in zip(maps, deltas):
                    map.volume -= delta

                    if map.min is not None or map.max is not None:
                        map.volume.clamp_(map.min, map.max)
                del deltas

                # --- compute GN gain ----------------------------------
                ll_gn_prev = ll_gn
                ll_gn = crit + reg + vreg + sumrls
                gain = ll_gn_prev - ll_gn
                if opt.verbose:
                    pstr = f'{n_iter_rls:3d} | {n_iter_prm:3d} | '
                    if opt.distortion.enable:
                        pstr += f'{"----":4s} | '
                        pstr += f'{"-"*72:72s} | '
                    else:
                        pstr += f'{"prm":4s} | '
                        pstr += f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} '
                        pstr += f'= {ll_gn:12.6g} | '
                    pstr += f'gain = {gain/ll_scl:7.2g}'
                    if opt.optim.nb_levels > 1:
                        pstr = f'{level:3d} | ' + pstr
                    print(pstr)
                    if opt.plot:
                        _show_maps(maps, dist, data)
                if gain < opt.optim.tolerance * ll_scl:
                    break

            # --------------------------------------------------------------
            #    Update RLS weights
            # --------------------------------------------------------------
            if level == 1 and opt.regularization.norm in ('tv', 'jtv'):
                reg = 0.5 * dot(maps.volume, regularizer(maps.volume))
                rls_changed = True
                sumrls_prev = sumrls
                rls, sumrls = reweight(maps.volume, return_sum=True)
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


def meetup(data, dist=None, opt=None):
    """Fit the ESTATICS+MEETUP model to multi-echo Gradient-Echo data.

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
        Observed GRE data.
    dist : sequence[Optional[ParameterizedDistortion]], optional
        Pre-computed distortion fields
    opt : Options, optional
        Algorithm options.

    Returns
    -------
    intecepts : sequence[GradientEcho]
        Echo series extrapolated to TE=0
    decay : estatics.ParameterMap
        R2* decay map
    distortions : sequence[ParameterizedDistortion]
        B0-induced distortion fields

    """
    # --- prepare everything -------------------------------------------
    data, maps, dist, opt, rls = _prepare(data, dist, opt)
    sumrls = 0.5 * rls.sum(dtype=torch.double) if rls is not None else 0
    backend = dict(dtype=opt.backend.dtype, device=opt.backend.device)

    # --- initialize tracking of the objective function ----------------
    crit, vreg, reg = float('inf'), 0, 0
    ll_rls = crit + vreg + reg
    sumrls_prev = sumrls
    rls_changed = False
    ll_scl = sum(core.py.prod(dat.shape) for dat in data)

    # --- Multi-Resolution loop ----------------------------------------
    shape0 = shape = maps.shape[1:]
    aff0 = aff = maps.affine
    lam0 = lam = opt.regularization.factor
    vx0 = vx = spatial.voxel_size(aff0)
    scl0 = vx0.prod()
    armijo_prev = 1
    for level in range(opt.optim.nb_levels, 0, -1):

        if opt.optim.nb_levels > 1:
            aff, shape = _get_level(level, aff0, shape0)
            vx = spatial.voxel_size(aff)
            scl = vx.prod() / scl0
            lam = [float(l * scl) for l in lam0]
            maps, rls = _resize(maps, rls, aff, shape)
            if opt.regularization.norm in ('tv', 'jtv'):
                sumrls = 0.5 * scl * rls.reciprocal().sum(dtype=torch.double)

        grad = torch.empty((len(data) + 1, *shape), **backend)
        hess = torch.empty((len(data) * 2 + 1, *shape), **backend)

        # --- RLS loop -------------------------------------------------
        max_iter_rls = 1 if level > 1 else opt.optim.max_iter_rls
        for n_iter_rls in range(1, max_iter_rls + 1):

            # --- helpers ----------------------------------------------
            regularizer_prm = lambda x: spatial.regulariser(
                x, weights=rls, dim=3,
                voxel_size=vx, membrane=1, factor=lam)
            solve_prm = lambda H, g: spatial.solve_field(
                H, g, rls, factor=lam,
                membrane=1 if opt.regularization.norm else 0,
                voxel_size=vx, max_iter=opt.optim.max_iter_cg,
                tolerance=opt.optim.tolerance_cg, dim=3)
            reweight = lambda x, **k: spatial.membrane_weights(
                x, lam, vx,  dim=3, **k,
                joint=opt.regularization.norm == 'jtv',
                eps=core.constants.eps(rls.dtype))

            # ----------------------------------------------------------
            #    Initial update of parameter maps
            # ----------------------------------------------------------
            crit_pre_prm = None
            max_iter_prm = opt.optim.max_iter_prm
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
                    gind, hind = [i, -1], [i, len(grad) - 1, len(grad) + i]
                    grad[gind] += g1
                    hess[hind] += h1
                    crit += crit1
                del g1, h1

                # --- regularization -----------------------------------
                reg = 0.
                if opt.regularization.norm:
                    g1 = regularizer_prm(maps.volume)
                    reg = 0.5 * dot(maps.volume, g1)
                    grad += g1
                    del g1

                if n_iter_prm == 1:
                    crit_pre_prm = crit

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
                    pstr = (f'{n_iter_rls-1:3d} | {"---":3s} | {"rls":4s} | '
                            f'{crit_prev:12.6g} + {reg:12.6g} + '
                            f'{sumrls:12.6g} + {vreg:12.6g} '
                            f'= {ll_tmp:12.6g} | {evol}')
                    if opt.optim.nb_levels > 1:
                        pstr = f'{level:3d} | ' + pstr
                    print(pstr)

                # --- gauss-newton -------------------------------------
                # Computing the GN step involves solving H\g
                hess = check_nans_(hess, warn='hessian')
                hess = hessian_loaddiag_(hess, 1e-6, 1e-8)
                deltas = solve_prm(hess, grad)
                deltas = check_nans_(deltas, warn='delta')

                dd = regularizer_prm(deltas)
                dv = dot(dd, maps.volume)
                dd = dot(dd, deltas)
                delta_reg = 0.5 * (dd - 2 * dv)

                for map, delta in zip(maps, deltas):
                    map.volume -= delta
                    if map.min is not None or map.max is not None:
                        map.volume.clamp_(map.min, map.max)
                del deltas

                # --- track parameter map improvement --------------
                gain = (crit_prev + reg_prev) - (crit + reg)
                if n_iter_prm > 1 and gain < opt.optim.tolerance * ll_scl:
                    break

            # ----------------------------------------------------------
            #    Distortion update with line search
            # ----------------------------------------------------------
            max_iter_dist = opt.optim.max_iter_dist
            for n_iter_dist in range(1, max_iter_dist + 1):

                crit = 0
                vreg = 0
                new_crit = 0
                new_vreg = 0

                deltas, dd, dv = [], 0, 0
                # --- loop over contrasts ------------------------------
                for i, (contrast, intercept, distortion) \
                        in enumerate(zip(data, maps.intercepts, dist)):

                    # --- helpers --------------------------------------
                    vxr = distortion.voxel_size[contrast.readout]
                    lam_dist = dict(opt.distortion.factor)
                    lam_dist['factor'] *= vxr ** 2
                    regularizer_dist = lambda x: spatial.regulariser(
                        x[None], **lam_dist, dim=3, bound=DIST_BOUND,
                        voxel_size=distortion.voxel_size)[0]
                    solve_dist = lambda h, g: spatial.solve_field_fmg(
                        h[None], g[None], **lam_dist, dim=3, bound=DIST_BOUND,
                        voxel_size=distortion.voxel_size)[0]

                    # --- likelihood -----------------------------------
                    crit1, g, h = derivatives_distortion(
                        contrast, distortion, intercept, maps.decay, opt)
                    crit += crit1

                    # --- regularization -------------------------------
                    vol = distortion.volume
                    g1 = regularizer_dist(vol)
                    vreg1 = 0.5 * vol.flatten().dot(g1.flatten())
                    vreg += vreg1
                    g += g1
                    del g1

                    # --- gauss-newton ---------------------------------
                    h = check_nans_(h, warn='hessian (distortion)')
                    h = hessian_loaddiag_(h[None], 1e-32, 1e-32, sym=True)[0]
                    delta = solve_dist(h, g)
                    delta = check_nans_(delta, warn='delta (distortion)')
                    deltas.append(delta)
                    del g, h

                    deltas.append(delta)
                    dd1 = regularizer_dist(delta)
                    dv += dot(dd1, vol)
                    dd1 = dot(dd1, delta)
                    dd += dd1
                    del delta, vol

                # --- track parameters improvement ---------------------
                if opt.verbose and n_iter_dist == 1:
                    gain = crit_pre_prm - (crit + delta_reg)
                    evol = '<=' if gain > 0 else '>'
                    ll_tmp = crit + reg + vreg + sumrls
                    pstr = (f'{n_iter_rls:3d} | {"---":3} | {"prm":4s} | '
                            f'{crit:12.6g} + {reg + delta_reg:12.6g} + '
                            f'{sumrls:12.6g} + {vreg:12.6g} '
                            f'= {ll_tmp:12.6g} | '
                            f'gain = {gain / ll_scl:7.2g} | {evol}')
                    if opt.optim.nb_levels > 1:
                        pstr = f'{level:3d} | ' + pstr
                    print(pstr)

                # --- line search ----------------------------------
                reg = reg + delta_reg
                new_crit, new_reg = crit, reg
                armijo, armijo_prev, ok = armijo_prev, 0, False
                maps0 = maps
                for n_ls in range(1, 12+1):
                    for delta1, dist1 in zip(deltas, dist):
                        dist1.volume.sub_(delta1, alpha=(armijo - armijo_prev))
                    armijo_prev = armijo
                    new_vreg = 0.5 * armijo * (armijo * dd - 2 * dv)

                    maps = maps0.deepcopy()
                    max_iter_prm = opt.optim.max_iter_prm
                    for n_iter_prm in range(1, max_iter_prm + 1):

                        crit_prev = new_crit
                        reg_prev = new_reg

                        # --- loop over contrasts ------------------
                        new_crit = 0
                        grad.zero_()
                        hess.zero_()
                        for i, (contrast, intercept, distortion) in enumerate(zip(data, maps.intercepts, dist)):
                            # compute gradient
                            new_crit1, g1, h1 = derivatives_parameters(
                                contrast, distortion, intercept,
                                maps.decay, opt)
                            # increment
                            gind, hind = [i, -1], [i, len(grad) - 1,
                                                   len(grad) + i]
                            grad[gind] += g1
                            hess[hind] += h1
                            new_crit += new_crit1
                        del g1, h1

                        # --- regularization -----------------------
                        new_reg = 0.
                        if opt.regularization.norm:
                            g1 = regularizer_prm(maps.volume)
                            new_reg = 0.5 * dot(maps.volume, g1)
                            grad += g1
                            del g1

                        new_gain = (crit_prev + reg_prev) - (new_crit + new_reg)
                        if new_gain < opt.optim.tolerance * ll_scl:
                            break

                        # --- gauss-newton -------------------------
                        # Computing the GN step involves solving H\g
                        hess = check_nans_(hess, warn='hessian')
                        hess = hessian_loaddiag_(hess, 1e-6, 1e-8)
                        delta = solve_prm(hess, grad)
                        delta = check_nans_(delta, warn='delta')

                        dd = regularizer_prm(delta)
                        dv = dot(dd, maps.volume)
                        dd = dot(dd, delta)
                        delta_reg = 0.5 * (dd - 2 * dv)

                        for map, delta1 in zip(maps, delta):
                            map.volume -= delta1
                            if map.min is not None or map.max is not None:
                                map.volume.clamp_(map.min, map.max)
                        del delta

                    if new_crit + new_reg + new_vreg <= crit + reg:
                        ok = True
                        break
                    else:
                        armijo = armijo / 2

                if not ok:
                    for delta1, dist1 in zip(deltas, dist):
                        dist1.volume.add_(delta1, alpha=armijo_prev)
                    armijo_prev = 1
                    maps = maps0
                    new_crit = crit
                    new_vreg = 0
                    del delta
                else:
                    armijo_prev *= 1.5
                new_vreg = vreg + new_vreg

                # --- track distortion improvement ---------------------
                if not ok:
                    evol = '== (x)'
                elif new_crit + new_vreg <= crit + vreg:
                    evol = f'<= ({n_ls:2d})'
                else:
                    evol = f'> ({n_ls:2d})'
                gain = (crit + vreg + reg) - (new_crit + new_vreg + new_reg)
                crit, reg, reg = new_crit, new_vreg, new_reg
                if opt.verbose:
                    ll_tmp = crit + reg + vreg + sumrls
                    pstr = (
                        f'{n_iter_rls:3d} | {n_iter_dist:3d} | {"dist":4s} | '
                        f'{crit:12.6g} + {reg:12.6g} + {sumrls:12.6g} ')
                    pstr += f'+ {vreg:12.6g} '
                    pstr += f'= {ll_tmp:12.6g} | gain = {gain / ll_scl:7.2g} | '
                    pstr += f'{evol}'
                    if opt.optim.nb_levels > 1:
                        pstr = f'{level:3d} | ' + pstr
                    print(pstr)
                    if opt.plot:
                        _show_maps(maps, dist, data)
                if not ok:
                    break

                if n_iter_dist > 1 and gain < opt.optim.tolerance * ll_scl:
                    break

            # --------------------------------------------------------------
            #    Update RLS weights
            # --------------------------------------------------------------
            if level == 1 and opt.regularization.norm in ('tv', 'jtv'):
                rls_changed = True
                sumrls_prev = sumrls
                rls, sumrls = reweight(maps.volume, return_sum=True)
                sumrls = 0.5 * sumrls

                # --- compute gain -----------------------------------------
                # (we are late by one full RLS iteration when computing the
                #  gain but we save some computations)
                ll_rls_prev = ll_rls
                ll_rls = crit + reg + vreg
                gain = ll_rls_prev - ll_rls
                if gain < opt.optim.tolerance * ll_scl:
                    print(f'RLS converged ({gain / ll_scl:7.2g})')
                    break

    # --- prepare output -----------------------------------------------
    out = postproc(maps, data)
    if opt.distortion.enable:
        out = (*out, dist)
    return out


@torch.jit.script
def recon_fit(inter, slope, te: float):
    """Reconstruct a single echo"""
    return inter.add(slope, alpha=-te).exp()


def derivatives_parameters(contrast, distortion, intercept, decay, opt,
                           do_grad=True):
    """Compute the gradient and Hessian of the parameter maps with
    respect to one contrast.

    Parameters
    ----------
    contrast : (nb_echo, *obs_shape) GradientEchoMulti
        A single echo series (with the same weighting)
    distortion : ParameterizedDistortion
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

        # forward model
        dat = echo.fdata(**backend, rand=True, missing=0)
        fit = recon_fit(inter, slope, te)
        push_fit, _ = pull1d(fit, grid_blip, readout)
        if jac_blip is not None:
            push_fit = push_fit * jac_blip
        msk = get_mask_missing(dat, push_fit)
        dat.masked_fill_(msk, 0)
        push_fit.masked_fill_(msk, 0)
        msk = msk.bitwise_not_()

        # likelihood / residuals
        nll = nll_chi if chi else nll_gauss
        opt = [df] if chi else []
        crit1, res = nll(dat, push_fit, msk, lam, *opt)
        del dat, push_fit
        crit += crit1

        if do_grad:
            msk = msk.to(fit.dtype)
            if grid_blip is not None:
                res *= jac_blip
                abs_res = res
                res = push1d(res, grid_blip, readout)
                abs_res.mul_(jac_blip).abs_()
                abs_res = push1d(abs_res, grid_blip, readout)
                abs_res.mul_(fit)
                msk = msk.mul_(jac_blip).mul_(jac_blip)
                msk = push1d(msk, grid_blip, readout)

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

    crit = 0
    grad = torch.zeros(obs_shape, **backend) if do_grad else None
    hess = torch.zeros(obs_shape, **backend) if do_grad else None

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
        fit, gfit = pull1d(fit, grid_blip, readout, grad=do_grad)
        fit *= jac_blip
        msk = get_mask_missing(dat, fit)    # mask of missing values
        if do_grad:
            gfit *= jac_blip
            gfit.masked_fill_(msk, 0)
        dat.masked_fill_(msk, 0)
        fit.masked_fill_(msk, 0)
        msk = msk.bitwise_not_()

        nll = nll_chi if chi else nll_gauss
        opt = [df] if chi else []
        crit1, res = nll(dat, fit, msk, lam, *opt)
        del dat, msk
        crit += crit1

        if do_grad:
            # if Dense:
            #   g1 = res * jac * grad + div(res * fit)
            #   h1 = (jac * grad)**2 + div(div(fit ** 2))
            # if SVF:
            #   g1 = res * jac**2 * grad + div(res * jac * fit)
            #   h1 = (jac**2 * grad)**2 + div(div((jac * fit)**2))
            if isinstance(distortion, SVFDistortion):
                gfit *= jac_blip
            else:
                fit /= jac_blip
            g1 = div1d(res * fit, readout)
            g1.addcmul_(res, gfit)
            # h1 = div1d(div1d(fit * fit, readout), readout)
            # h1.addcmul_(gfit, gfit)
            h1 = gfit.square()

            # propagate backward
            if isinstance(distortion, SVFDistortion):
                vel = distortion.volume
                if blip < 0:  vel = -vel
                g1, h1 = spatial.exp1d_backward(
                    vel, g1, h1, steps=distortion.steps, bound=DIST_BOUND)

            # accumulate
            alpha_g = alpha_h = lam
            alpha_g = alpha_g * blip
            grad.add_(g1, alpha=alpha_g)
            hess.add_(h1, alpha=alpha_h)

    if not do_grad:
        return crit
    mask_nan_(grad)
    mask_nan_(hess)
    return crit, grad, hess


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
            vol = dst.volume
            plt.subplot(1 + has_dist, ncol, i+1+ncol)
            vol = vol[:, :, dst.shape[-2]//2]
            plt.imshow(vol.cpu(), vmin=-2, vmax=2)
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