import torch
from nitorch import core
from ._options import ESTATICSOptions
from ._preproc import preproc, postproc
from ._utils import hessian_loaddiag_, hessian_solve
from ..utils import smart_grid, smart_pull, smart_push


def loglin(data, opt=None):
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

    opt = ESTATICSOptions().update(opt)

    # --- estimate noise / register / initialize maps ---
    data, maps = preproc(data, opt)

    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)
    mean_shape = maps.decay.volume.shape

    # --- compute derivatives ---
    grad = torch.empty((len(data) + 1,) + mean_shape, **backend)
    hess = torch.empty((len(data) * 2 + 1,) + mean_shape, **backend)

    for n_iter in range(opt.optim.max_iter_gn):
        decay = maps.decay

        grad.zero_()
        hess.zero_()
        crit = 0

        # --- loop over contrasts ---
        for i, (contrast, intercept) in enumerate(zip(data, maps.intercepts)):

            crit1, g1, h1 = _loglin_gradient(contrast, intercept, decay, opt)

            # increment
            gind = [i, -1]
            grad[gind, ...] += g1
            hind = [i, len(grad)-1, len(grad)+i]
            hess[hind, ...] += h1
            crit += crit1

        print('{:3d} | {:12.6g}'.format(n_iter, crit))

        # --- mask of voxels where there's no data ---
        msk = hess[:len(grad)] == 0

        # --- load diagonal of the Hessian ---
        hess = hessian_loaddiag_(hess, 1e-6, 1e-8)
        
        # --- gauss-newton ---
        deltas = hessian_solve(hess, grad)

        for map, delta in zip(maps, deltas):
            map.volume -= delta
            if map.min is not None or map.max is not None:
                map.volume.clamp_(map.min, map.max)

    crit = sum(_loglin_gradient(contrast, intercept, maps.decay, opt, do_grad=False)
               for contrast, intercept in zip(data, maps.intercepts))
    print('{:3d} | {:12.6g}'.format(n_iter+1, crit))

    te0, decay = postproc(maps, data)
    for d, map in enumerate(te0):
        map.volume[msk[d]] = 0  # replace voxels where there's no data
    return te0, decay


def _loglin_gradient(contrast, intercept, decay, opt, do_grad=True):
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
    inter_slope = smart_pull(inter_slope, grid)
    inter, slope = inter_slope

    crit = 0
    grad = torch.zeros((2,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((3,) + obs_shape, **backend) if do_grad else None

    for echo in contrast:

        # compute residuals
        dat = echo.fdata(**backend, cache=False)             # observed
        fit = inter - echo.te * slope                        # fitted
        msk = torch.isfinite(fit) & torch.isfinite(dat) & (dat > 0)    # mask of observed
        dat = dat.log_()
        dat[~msk] = 0
        fit[~msk] = 0
        res = fit
        res -= dat                                           # residuals
        del dat, fit
        msk = msk.to(**backend)

        # compute log-likelihood
        crit = crit - 0.5 * lam * res.square().sum(dtype=torch.double)

        if do_grad:
            # compute gradient and Hessian in observed space
            grad1 = lam * res
            grad[0] += grad1
            grad1 *= echo.te
            grad[1] -= grad1
            del res, grad1
            hess1 = lam * msk
            hess[0] += hess1
            hess1 *= echo.te
            hess[2] -= hess1
            hess1 *= echo.te
            hess[1] += hess1
            del hess1

    if do_grad:
        # push gradient and Hessian to recon space
        grad = smart_push(grad, grid, recon_shape)
        hess = smart_push(hess, grid, recon_shape)
        return crit, grad, hess

    return crit
