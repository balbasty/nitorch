import torch
import nitorch as ni
from nitorch import core, spatial
from ._options import Options
from ._preproc import preproc, postproc
from ._utils import (hessian_sym_loaddiag, hessian_sym_matmul, hessian_sym_solve,
                     smart_grid, smart_pull, smart_push)
from ..param import ParameterMap


def nonlin(data, transmit=[], receive=[], opt=None):
    """Fit a non-linear relaxometry model to multi-echo Gradient-Echo data.

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
        Observed GRE data.
    transmit : sequence[PrecomputedFieldMap], optional
        Map(s) of the transmit field (b1+). If a single map is provided,
        it is used to correct all contrasts. If multiple maps are
        provided, there should be one for each contrast.
    receive : sequence[PrecomputedFieldMap], optional
        Map(s) of the receive field (b1-). If a single map is provided,
        it is used to correct all contrasts. If multiple maps are
        provided, there should be one for each contrast.
        If no receive map is provided, the output `pd` map will have
        a remaining b1- bias field.
    opt : Options, optional
        Algorithm options.

    Returns
    -------
    intecepts : sequence[GradientEcho]
        Echo series extrapolated to TE=0
    decay : estatics.ParameterMap
        R2* decay map

    """

    if opt is None:
        opt = Options()
    opt = opt.copy()
    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    # --- estimate noise / register / initialize maps ---
    data, transmit, receive, maps = preproc(data, transmit, receive, opt)
    vx = spatial.voxel_size(maps.affine)
    has_mt = hasattr(maps, 'mt')

    # --- prepare regularization factor ---
    lam = opt.regularization.factor
    lam = core.utils.make_list(lam, 3 + has_mt)  # PD, R1, R2*, MT

    # --- initialize weights (RLS) ---
    if (not opt.regularization.norm or
            opt.regularization.norm.lower() == 'none' or
            all(l == 0 for l in lam)):
        opt.regularization.norm = ''
    opt.regularization.norm = opt.regularization.norm.lower()
    mean_shape = maps[0].shape
    rls = None
    sumrls = 0
    if opt.regularization.norm in ('tv', 'jtv'):
        rls_shape = mean_shape
        if opt.regularization.norm == 'tv':
            rls_shape = (len(maps),) + rls_shape
        rls = ParameterMap(rls_shape, fill=1, **backend).volume
        sumrls = rls.sum(dtype=torch.double)
    multi_rls = rls if opt.regularization.norm == 'tv' \
                else [rls] * len(maps)

    if opt.regularization.norm:
        print(f'With {opt.regularization.norm.upper()} regularization:')
        print(f'    - PD:  {lam[0]:.3g}')
        print(f'    - R1:  {lam[1]:.3g}')
        print(f'    - R2*: {lam[2]:.3g}')
        if has_mt:
            print(f'    - MT:  {lam[3]:.3g}')
    else:
        print('Without regularization:')

    if opt.regularization.norm not in ('tv', 'jtv'):
        # no reweighting -> do more gauss-newton updates instead
        opt.optim.max_iter_gn *= opt.optim.max_iter_rls
        opt.optim.max_iter_rls = 1

    if opt.verbose:
        print('{:^3s} | {:^3s} | {:^12s} + {:^12s} + {:^12s} = {:^12s} | {:^2s} | {:^7s}'
              .format('rls', 'gn', 'fit', 'reg', 'rls', 'crit', 'ls', 'gain'))

    shape0 = shape = maps.shape
    aff0 = aff = maps.affine
    for level in range(opt.optim.nb_levels, 0, -1):
        
        if opt.optim.nb_levels > 1:
            print('begin iter')
            show_maps(maps)
            aff, shape = _get_level(level, aff0, shape0)
            maps, rls = _resize(maps, rls, aff, shape)
            print('after resize')
            show_maps(maps)
        
        # --- compute derivatives ---
        nb_prm = len(maps)
        nb_hes = nb_prm * (nb_prm + 1) // 2
        grad = torch.empty((nb_prm,) + shape, **backend)
        hess = torch.empty((nb_hes,) + shape, **backend)
    
        ll_rls = []
        ll_max = core.constants.ninf

        for n_iter_rls in range(opt.optim.max_iter_rls):

            multi_rls = rls if opt.regularization.norm == 'tv' \
                        else [rls] * len(maps)

            # --- Gauss Newton loop ---
            ll_gn = []
            for n_iter_gn in range(opt.optim.max_iter_gn):
                crit = 0
                grad.zero_()
                hess.zero_()

                # --- loop over contrasts ---
                for contrast, b1m, b1p in zip(data, receive, transmit):
                    # compute gradient
                    crit1, g1, h1 = _nonlin_gradient(contrast, maps, b1m, b1p, opt)

                    # increment
                    if hasattr(maps, 'mt') and not contrast.mt:
                        grad[:-1] += g1
                        hind = list(range(nb_prm-1))
                        cnt = nb_prm
                        for i in range(nb_prm):
                            for j in range(i+1, nb_prm):
                                if i != nb_prm-1 and j != nb_prm-1:
                                    hind.append(cnt)
                                cnt += 1
                        hess[hind] += h1
                        crit += crit1
                    else:
                        grad += g1
                        hess += h1
                        crit += crit1

                # --- regularization ---
                reg = 0.
                if opt.regularization.norm:
                    for i, (map, weight, l) in enumerate(zip(maps, multi_rls, lam)):
                        if not l:
                            continue
                        reg1, g1 = _nonlin_reg(map.fdata(), vx, weight, l)
                        reg += reg1
                        grad[i] += g1

                # --- load diagonal of the Hessian ---
                hess = hessian_sym_loaddiag(hess)

                # --- gauss-newton ---
                if opt.regularization.norm:
                    deltas = _nonlin_solve(hess, grad, multi_rls, lam, vx, opt)
                else:
                    deltas = hessian_sym_solve(hess, grad)

                if not opt.optim.max_iter_ls:
                    # --- check improvement ---
                    n_iter_ls = 0
                    for map, delta in zip(maps, deltas):
                        map.volume -= delta
                        if map.min is not None or map.max is not None:
                            map.volume.clamp_(map.min, map.max)
                else:
                    # --- line search ---
                    armijo = 1.
                    ok = False
                    crit0 = crit
                    reg0 = reg
                    maps0 = maps
                    if opt.verbose > 1:
                        print('{:3d} | {:3d} | {:12.6g} + {:12.6g} + {:12.6g} = {:12.6g} | {:2d}'
                              .format(n_iter_rls, n_iter_gn, crit, reg, sumrls, crit + reg + sumrls, 0))
                    for n_iter_ls in range(opt.optim.max_iter_ls):
                        maps = maps0.deepcopy()
                        for map, delta in zip(maps, deltas):
                            map.volume -= armijo * delta
                            if map.min is not None or map.max is not None:
                                map.volume.clamp_(map.min, map.max)
                        crit = sum(_nonlin_gradient(contrast, maps, b1m, b1p, opt, do_grad=False)
                                   for contrast, b1m, b1p in zip(data, receive, transmit))
                        if opt.regularization.norm:
                            reg = sum(_nonlin_reg(map.volume, vx, weight, l, do_grad=False)
                                      for map, weight, l in zip(maps, multi_rls, lam))
                        else:
                            reg = 0.
                        if opt.verbose > 1:
                            print('{:3d} | {:3d} | {:12.6g} + {:12.6g} + {:12.6g} = {:12.6g} | {:2d}  {:2s}'
                                  .format(n_iter_rls, n_iter_gn, crit, reg, sumrls,
                                          crit + reg + sumrls, n_iter_ls+1,
                                          ':D' if ( crit + reg < crit0 + reg0) else ':('))
                        if crit + reg < crit0 + reg0:
                            ok = True
                            break
                        else:
                            armijo /= 2.
                    if not ok:
                        reg = reg0
                        crit = crit0
                        maps = maps0
                        break


                print('end iter')
                show_maps(maps)
                # --- Compute gain ---
                ll = crit + reg + sumrls
                ll_max = max(ll_max, ll)
                ll_prev = ll_gn[-1] if ll_gn else ll_max
                if ll_max != ll_prev:
                    gain = (ll_prev - ll) / (ll_max - ll_prev)
                else:
                    gain = core.constants.inf
                ll_gn.append(ll)
                if opt.verbose:
                    print('{:3d} | {:3d} | {:12.6g} + {:12.6g} + {:12.6g} = {:12.6g} | {:2d} | gain = {:7.2g}'
                          .format(n_iter_rls, n_iter_gn, crit, reg, sumrls,
                                  crit + reg + sumrls, n_iter_ls, gain))
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
                ll_prev = ll_rls[-1][-1] if ll_rls else ll_max
                ll_rls.append(ll_gn)
                gain = (ll_prev - ll) / (ll_max - ll_prev)
                if gain < opt.optim.tolerance_rls:
                    print(f'Converged ({gain:7.2g})')
                    break

    # --- Prepare output ---
    return postproc(maps)


def _get_level(level, aff0, shape0):
    return spatial.affine_resize(aff0, shape0, 1/(2 ** (level-1)))


def _resize(maps, rls, aff, shape):
    grid = spatial.affine_grid(aff, shape)[None, ...]
    for map in maps:
        map.volume = spatial.resize(map.volume[None, None, ...], 
                                    shape=shape)[0, 0]
        map.affine = aff
    maps.affine = aff
    maps.shape = shape
    if rls is not None:
        if rls.dim() == len(shape):
            rls = spatial.resize(rls[None, None], shape=shape)[0, 0]
        else:
            rls = spatial.resize(rls[None], shape=shape)[0]
    return maps, rls


def _nonlin_gradient(contrast, maps, receive, transmit, opt, do_grad=True):
    """Compute the gradient and Hessian of the parameter maps with
    respect to one contrast.

    Parameters
    ----------
    contrast : (nb_echo, *obs_shape) GradientEchoMulti
        A single echo series (with the same weighting)
    maps : (*recon_shape) ParameterMaps
        Current parameter maps, with fields pd, r1, r2s and mt.
    receive :
    transmit :
    opt : Options
    do_grad : bool, default=True
        If False, only compute the lnegative og-likelihood.

    Returns
    -------
    crit : () tensor
        Negative log-likelihood
    grad : (3|4, *recon_shape) tensor
        Gradient with respect to (the log of):
            [0] pd
            [1] r1
            [2] r2s
            [3] mt (optional)
    hess : (6|10, *recon_shape) tensor
        Hessian with respect to:
            [0] pd ** 2
            [1] r1 ** 2
            [2] r2s ** 2
            [ |3] mt ** 2 (optional)
            [3|4] pd * r1
            [4|5] pd * r2s
            [ |6] pd * mt (optional)
            [5|7] r1 * r2s
            [ |8] r1 * mt (optional)
            [ |9] r2s * mt (optional)

    """

    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    # sequence parameters
    lam = 1 / contrast.noise
    tr = contrast.tr
    fa = contrast.fa / 180. * core.constants.pi
    
    obs_shape = contrast.shape[1:]
    recon_shape = maps.shape

    # pull parameter maps to observed space
    aff = core.linalg.lmdiv(maps.affine, contrast.affine)
    aff = aff.to(**backend)
    grid = smart_grid(aff, obs_shape, recon_shape)
    dmaps = torch.stack([map.fdata(**backend) for map in maps])
    dmaps = smart_pull(dmaps, grid)
    if hasattr(maps, 'mt'):
        pd, r1, r2s, mt = dmaps
    else:
        pd, r1, r2s = dmaps
    has_mt = bool(contrast.mt)
    if has_mt:
        nb_prm = 4
        nb_hes = 10
    else:
        nb_prm = 3
        nb_hes = 6

    # pull field maps to observed space
    if receive is not None:
        aff = core.linalg.lmdiv(receive.affine, contrast.affine)
        aff = aff.to(**backend)
        grid1 = smart_grid(aff, obs_shape, receive.shape)
        b1m = smart_pull(receive.fdata(**backend)[None, ...], grid1)[0]
        if receive.unit in ('%', 'pct', 'p.u.'):
            if grid1 is None:
                b1m = b1m * 100
            else:
                b1m /= 100
        del grid1

    if transmit is not None:
        aff = core.linalg.lmdiv(transmit.affine, contrast.affine)
        aff = aff.to(**backend)
        grid1 = smart_grid(aff, obs_shape, transmit.shape)
        b1p = smart_pull(transmit.fdata(**backend)[None, ...], grid1)[0]
        if grid1 is None:
            b1p = b1p.clone()
        if transmit.unit in ('%', 'pct', 'p.u.'):
            b1p /= 100
        del grid1

    # exponentiate
    pd = pd.exp() if grid is not None else pd.exp_()
    r1 = r1.exp() if grid is not None else r1.exp_()
    r2s = r2s.exp() if grid is not None else r2s.exp_()
    if has_mt:
        # mt is encoded by a sigmoid:
        # > mt = 1 / (1 + exp(-prm))
        mt = mt.neg() if grid is not None else mt.neg_()
        mt = mt.exp_()
        mt += 1
        mt = mt.reciprocal_()

    # precompute intercept
    #                             (1 - mt) * (1 - exp(-r1*tr))
    # fit0 = pd * sin(fa) * ----------------------------------------
    #                        (1 - cos(fa) * (1 - mt) * exp(-r1*tr))
    fit0 = pd
    if receive is not None:
        fit0 *= b1m
        del b1m
    if transmit:
        b1p *= fa
        fa = b1p
    else:
        fa = torch.as_tensor(fa, **backend)
    fit0 *= fa.sin()
    cosfa = fa.cos_()
    e1 = r1
    e1 *= -tr
    e1 = e1.exp_()
    fit0 *= (1 - e1)
    if has_mt:
        omt = mt.neg_()
        omt += 1
        fit0 *= omt
        fit0 /= (1 - cosfa * omt * e1)
    else:
        fit0 /= (1 - cosfa * e1)

    crit = 0
    grad = torch.zeros((nb_prm,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((nb_hes,) + obs_shape, **backend) if do_grad else None

    for echo in contrast:

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)  # observed
        fit = fit0 * (-echo.te * r2s).exp_()                 # fitted
        msk = fit.isfinite() & dat.isfinite() & (dat > 0)    # mask of observed
        dat[~msk] = 0
        fit[~msk] = 0
        res = dat.neg_()
        res += fit
        del dat

        # compute log-likelihood
        crit = crit + 0.5 * lam * res.square().sum(dtype=torch.double)

        if do_grad:

            # PD / R1 / R2* / MT
            grad1 = torch.empty_like(grad)

            # compute gradient of the signal term
            # (grad_crit = grad_signal * residuals)
            grad1[0] = fit
            grad1[2] = -echo.te * r2s * fit
            if has_mt:
                grad1[3] = fit
                grad1[3] *= (omt - 1) / (1 - omt * cosfa * e1)
                omt_x_cosfa = omt * cosfa
            else:
                omt_x_cosfa = cosfa
            grad1[1] = -tr * r1 * fit * (omt_x_cosfa - 1) * e1
            grad1[1] /= (1 - e1) * (1 - omt_x_cosfa * e1)

            # compute diagonal of the hessian of the signal term 
            # (hess_crit = grad_signal^2 + hess_signal * residuals)
            hess0 = torch.empty_like(grad)
            hess0[0] = grad1[0]
            hess0[2] = grad1[2]
            hess0[2] *= (1 - echo.te * r2s)
            if has_mt:
                omt_x_cosfa = omt * cosfa
                hess0[3] = (omt - 1) * omt_x_cosfa * e1 
                hess0[3] /= 1 - omt_x_cosfa * e1
                hess0[3] += omt 
                hess0[3] -= 0.5
                hess0[3] *= 2
                hess0[3] = grad1[3]
            else:
                omt_x_cosfa = cosfa
            hess0[1] = - tr * r1 
            hess0[1] *= (1 + omt_x_cosfa * e1) 
            hess0[1] /= (1 - omt_x_cosfa * e1)
            hess0[1] += 1
            hess0[1] *= grad1[1]
            
            hess0 *= res
            hess0 = hess0.abs_()
            
            # increment hessian
            hess1 = torch.empty_like(hess)
            hess1[0] = grad1[0].square() + hess0[0]
            hess1[1] = grad1[1].square() + hess0[1]
            hess1[2] = grad1[2].square() + hess0[2]
            if has_mt:
                hess1[3] = grad1[3].square() + hess0[3]
                del hess0
                hess1[4] = grad1[0] * grad1[1]
                hess1[5] = grad1[0] * grad1[2]
                hess1[6] = grad1[0] * grad1[3]
                hess1[7] = grad1[1] * grad1[2]
                hess1[8] = grad1[1] * grad1[3]
                hess1[9] = grad1[2] * grad1[3]
            else:
                del hess0
                hess1[3] = grad1[0] * grad1[1]
                hess1[4] = grad1[0] * grad1[2]
                hess1[5] = grad1[1] * grad1[2]
            hess1 *= lam
            hess1[:, ~msk] = 0
            hess += hess1
            del hess1

            # increment gradient
            grad1 *= res
            grad1 *= lam
            grad += grad1
            del grad1

    if do_grad:
        # push gradient and Hessian to recon space
        grad = smart_push(grad, grid, recon_shape)
        hess = smart_push(hess, grid, recon_shape)
        return crit, grad, hess

    return crit


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

    grad_fwd = spatial.diff(map, dim=[0, 1, 2], voxel_size=vx, side='f')
    grad_bwd = spatial.diff(map, dim=[0, 1, 2], voxel_size=vx, side='b')
    if rls is not None:
        grad_fwd *= rls[..., None]
        grad_bwd *= rls[..., None]
    grad_fwd = spatial.div(grad_fwd, dim=[0, 1, 2], voxel_size=vx, side='f')
    grad_bwd = spatial.div(grad_bwd, dim=[0, 1, 2], voxel_size=vx, side='b')

    grad = grad_fwd
    grad += grad_bwd
    grad *= lam / 2.   # average across directions (3) and side (2)

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
    hess : (P*(P+1)//2, *shape) tensor
    grad : (P, *shape) tensor
    rls : (P, *shape) tensor_like
    lam : (P,) sequence[float]
    vx : (D,) sequence[float]
    opt : Options

    Returns
    -------
    delta : (P, *shape) tensor

    """

    def hess_fn(x):
        result = hessian_sym_matmul(hess, x)
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
    # We use (H + W*diag(D'D)) as a preconditioner because it is easy to
    # invert. I think that it works because D'D has a nice form where
    # its rows sum to zero. Otherwise, we'd need to add a bit of
    # something on the diagonal of the preconditioner.
    # Furthermore, diag(D'D) = d*I, where d is the central weight in
    # the corresponding convolution
    hessp = hess.clone()
    smo = 2 * torch.as_tensor(vx).square().reciprocal().sum().item()
    for i, (weight, l) in enumerate(zip(rls, lam)):
        hessp[i] += l * smo * (weight if weight is not None else 1)

    def precond(x):
        return hessian_sym_solve(hessp, x)

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
        grad_fwd = spatial.diff(maps.fdata(), dim=[0, 1, 2], voxel_size=vx, side='f')
        grad_bwd = spatial.diff(maps.fdata(), dim=[0, 1, 2], voxel_size=vx, side='b')

        grad = grad_fwd.square_().sum(-1)
        grad += grad_bwd.square_().sum(-1)
        grad *= lam / 2.   # average across directions (3) and side (2)
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


def show_maps(maps):
    import matplotlib.pyplot as plt
    clamps = [2e4, 1.5, 50, 3]
    n = 3 + hasattr(maps, 'mt')
    z = maps.shape[1]//2
    plt.subplot(1, n, 1)
    q = plt.imshow(maps.pd.fdata()[:, z, :].exp().cpu().clamp_max(2e4))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(1, n, 2)
    plt.imshow(maps.r1.fdata()[:, z, :].exp().cpu().clamp_max(1.5))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(1, n, 3)
    plt.imshow(maps.r2s.fdata()[:, z, :].exp().cpu().clamp_max(50))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    if hasattr(maps, 'mt'):
        plt.subplot(1, n, 4)
        mtslice = 100 / (1 + maps.mt.fdata()[:, z, :].neg().exp())
        plt.imshow(mtslice.cpu().clamp_max(3))
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
    plt.show()
    
def show_images(*args):
    clamps = [2e4, 1.5, 50, 3]
    import matplotlib.pyplot as plt
    z = args[0].shape[1]//2
    for i, arg in enumerate(args):
        plt.subplot(1, len(args), i+1)
        plt.imshow(arg[:, z, :].cpu())
        plt.colorbar()
        plt.axis('off')
    plt.show()
