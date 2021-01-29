import torch
from nitorch import core, spatial
from ._options import ESTATICSOptions
from ._preproc import preproc, postproc
from ._utils import (hessian_loaddiag, hessian_matmul, hessian_solve,
                     smart_grid, smart_pull, smart_push)
from ..utils import rls_maj
from nitorch.tools.qmri.param import ParameterMap


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

    """

    if opt is None:
        opt = ESTATICSOptions()
    opt = opt.copy()
    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    # --- be polite ---
    print(f'Fitting a (multi) exponential decay model with {len(data)} contrasts. Echo times:')
    for i, contrast in enumerate(data):
        print(f'    - contrast {i:2d}: [' + ', '.join([f'{te*1e3:.1f}' for te in contrast.te]) + '] ms')
    
    # --- estimate noise / register / initialize maps ---
    data, maps = preproc(data, opt)
    vx = spatial.voxel_size(maps.affine)

    # --- prepare regularization factor ---
    lam = opt.regularization.factor
    lam = core.utils.make_list(lam)
    if len(lam) > 1:
        *lam, lam_decay = lam
    else:
        lam_decay = lam[0]
    lam = core.utils.make_list(lam, len(maps)-1)
    lam.append(lam_decay)

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
        print('{:^3s} | {:^3s} | {:^12s} + {:^12s} + {:^12s} = {:^12s}'
              .format('rls', 'gn', 'fit', 'reg', 'rls', 'crit'))

    ll_rls = []
    ll_max = core.constants.ninf
        
    for n_iter_rls in range(opt.optim.max_iter_rls):

        multi_rls = rls if opt.regularization.norm == 'tv' \
                    else [rls] * len(maps)

        # --- Gauss Newton loop ---
        ll_gn = []
        for n_iter_gn in range(opt.optim.max_iter_gn):
            decay = maps.decay
            crit = 0
            grad.zero_()
            hess.zero_()

            # --- loop over contrasts ---
            for i, (contrast, intercept) in enumerate(zip(data, maps.intercepts)):
                # compute gradient
                crit1, g1, h1 = _nonlin_gradient(contrast, intercept, decay, opt)

                # increment
                gind = [i, -1]
                grad[gind, ...] += g1
                hind = [2*i, -1, 2*i+1]
                hess[hind, ...] += h1
                crit += crit1

            # --- regularization ---
            reg = 0.
            if opt.regularization.norm:
                for i, (map, weight, l) in enumerate(zip(maps, multi_rls, lam)):
                    if not l:
                        continue
                    reg1, g1 = _nonlin_reg(map.volume, vx, weight, l)
                    reg += reg1
                    grad[i] += g1

            # --- gauss-newton ---
            if not hess.isfinite().all():
                print('WARNING: NaNs in hess (??)')
            if opt.regularization.norm:
                hess = hessian_loaddiag(hess, 1e-5, 1e-8)
                deltas = _nonlin_solve(hess, grad, multi_rls, lam, vx, opt)
            else:
                hess = hessian_loaddiag(hess, 1e-3, 1e-6)
                deltas = hessian_solve(hess, grad)
            if not deltas.isfinite().all():
                print('WARNING: NaNs in delta (non stable Hessian)')

            for map, delta in zip(maps, deltas):
                map.volume -= delta
                if map.min is not None or map.max is not None:
                    map.volume.clamp_(map.min, map.max)

            # --- Compute gain ---
            ll = crit + reg + sumrls
            ll_max = max(ll_max, ll)
            ll_prev = ll_gn[-1] if ll_gn else ll_max
            gain = (ll_prev - ll) / (ll_max - ll_prev)
            ll_gn.append(ll)
            if opt.verbose:
                print('{:3d} | {:3d} | {:12.6g} + {:12.6g} + {:12.6g} = {:12.6g} | gain = {:7.2g}'
                      .format(n_iter_rls, n_iter_gn, crit, reg, sumrls,
                              crit + reg + sumrls, gain))
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
    return postproc(maps, data)


def _nonlin_gradient(contrast, intercept, decay, opt, do_grad=True):
    """Compute the gradient and Hessian of the parameter maps with
    respect to one contrast.

    Parameters
    ----------
    contrast : (nb_echo, *obs_shape) GradientEchoMulti
        A single echo series (with the same weighting)
    intercept : (*recon_shape) ParameterMap
        Log-intercept of the contrast
    decay : (*recon_shape) ParameterMap
        Exponential decay
    opt : Options

    Returns
    -------
    crit : () tensor
        Log-likelihood
    grad : (2, *recon_shape) tensor
        Gradient with respect to:
            [0] intercept
            [1] decay
    hess : (3, *recon_shape) tensor
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
    inter_slope = torch.stack([intercept.fdata(**backend),
                               decay.fdata(**backend)])
    inter, slope = smart_pull(inter_slope, grid)

    crit = 0
    grad = torch.zeros((2,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((3,) + obs_shape, **backend) if do_grad else None

    for echo in contrast:

        # compute residuals
        dat = echo.fdata(**backend, rand=True, cache=False)  # observed
        fit = (inter - echo.te * slope).exp_()               # fitted
        msk = fit.isfinite() & dat.isfinite() & (dat > 0)    # mask of observed
        dat[~msk] = 0
        fit[~msk] = 0
        res = dat.neg_()
        res += fit
        del dat, msk

        # compute log-likelihood
        crit = crit + 0.5 * lam * res.square().sum(dtype=torch.double)

        if do_grad:
            # compute gradient and Hessian in observed space
            #
            #   grad[inter]       =           lam * res * fit
            #   grad[decay]       =     -te * lam * res * fit
            #   hess[inter**2]    =           lam * fit * fit + (grad[inter] if res > 0)
            #   hess[decay**2]    = (te*te) * lam * fit * fit + (grad[decay] if res > 0)
            #   hess[inter*decay] =     -te * lam * fit * fit
            #
            # When (res < 0), the true Hessian is not positive definite
            # and the optimization problem is not convex
            # -> we use the expected Hessian instead.

            res *= fit
            res *= lam
            grad[0] += res
            grad[1] -= res * echo.te
            res = res.abs_()

            fit = fit.square_()
            fit *= lam
            hess[0] += fit
            hess[0] += res
            fit *= echo.te
            hess[2] -= fit
            fit *= echo.te
            hess[1] += fit
            res *= (echo.te * echo.te)
            hess[1] += res
            del res, fit

    
    grad[~torch.isfinite(grad)] = 0
    diag = hess[:-1]
    diag[~torch.isfinite(diag)] = 1e-3
    offdiag = hess[-1]
    offdiag[~torch.isfinite(offdiag)] = 0
            
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
    grad *= lam / 2   # average across side (2)

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
