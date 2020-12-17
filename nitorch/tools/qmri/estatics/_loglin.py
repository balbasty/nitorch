import torch
import nitorch as ni
from nitorch import core, spatial
from nitorch.tools.qmri import io as qio
from ._options import Options
from ._preproc import preproc, postproc
from ._utils import hessian_loaddiag,hessian_solve


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

    if opt is None:
        opt = Options()

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
            hind = [2*i, -1, 2*i+1]
            hess[hind, ...] += h1
            crit += crit1

        print('{:3d} | {:12.6g}'.format(n_iter, crit))

        # --- load diagonal of the Hessian ---
        hess = hessian_loaddiag(hess)

        # --- gauss-newton ---
        deltas = _loglin_solve(hess, grad)

        for map, delta in zip(maps, deltas):
            map.volume -= delta
            if map.min is not None or map.max is not None:
                map.volume.clamp_(map.min, map.max)

    crit = sum(_loglin_gradient(contrast, intercept, maps.decay, opt, do_grad=False)
               for contrast, intercept in zip(data, maps.intercepts))
    print('{:3d} | {:12.6g}'.format(n_iter+1, crit))

    return postproc(maps, data)


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
    grid = spatial.affine_grid(aff, obs_shape)[None, ...]
    inter_slope = torch.stack([intercept.fdata(**backend),
                               decay.fdata(**backend)])
    inter_slope = spatial.grid_pull(inter_slope[None, ...], grid)[0]
    inter, slope = inter_slope

    crit = 0
    grad = torch.zeros((2,) + obs_shape, **backend) if do_grad else None
    hess = torch.zeros((3,) + obs_shape, **backend) if do_grad else None

    for echo in contrast:

        # compute residuals
        dat = echo.fdata(**backend, cache=False)             # observed
        fit = inter - echo.te * slope                        # fitted
        msk = fit.isfinite() & dat.isfinite() & (dat > 0)    # mask of observed
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
        grad = spatial.grid_push(grad[None, ...], grid, recon_shape)[0]
        hess = spatial.grid_push(hess[None, ...], grid, recon_shape)[0]
        return crit, grad, hess

    return crit


def _loglin_solve(hess, grad):
    """Solve a batch of small linear systems

    Parameters
    ----------
    hess : (D*2+1, *shape) tensor
        Hessian
    grad : (D+1, *shape) tensor
        Gradient

    Returns
    -------
    diff : (D+1, *shape) tensor
        Solution

    """
    return hessian_solve(hess, grad)

    # # build full matrix
    # nb_prm = (hess.shape[0]-1)//2
    # nb_dim = hess.dim() - 1
    # hess0 = hess
    # zero = hess[0].new_zeros(1).expand(hess[0].shape)
    # hess = [[zero] * (nb_prm+1) for _ in range(nb_prm+1)]
    # for i in range(nb_prm):
    #     hess[i][i] = hess0[2*i]
    #     hess[i][-1] = hess0[2*i + 1]
    #     hess[-1][i] = hess0[2*i + 1]
    # hess[-1][-1] = hess0[-1]
    # del hess0
    # hess = core.utils.as_tensor(hess)
    #
    # # reorganize as batched matrices
    # hess = hess.permute(list(range(2, nb_dim+2)) + [0, 1])
    # grad = grad.permute(list(range(1, nb_dim+1)) + [0])
    #
    # # solve
    # grad, _ = torch.solve(grad[..., None], hess)
    # grad = grad[..., 0]
    # grad = grad.permute([-1] + list(range(nb_dim)))
    #
    # return grad
