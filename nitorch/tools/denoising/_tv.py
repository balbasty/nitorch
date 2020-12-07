import torch
from nitorch import core, spatial
from ._utils import hessian_matvec, first_to_last, last_to_first


# TODO: extend to multivariate Gaussian noise + missing data


def denoise(data, opt=None):
    """Fit the ESTATICS model to multi-echo Gradient-Echo data.

    Parameters
    ----------
    data : (P, ...) tensor_like
        Observed data
    opt : Options, optional
        Algorithm options.

    Returns
    -------
    denoised : (P, ...) tensor
        Denoised data

    """

    if opt is None:
        opt = Options()
    dtype = opt.backend.dtype
    device = opt.backend.device
    backend = dict(dtype=dtype, device=device)

    # --- estimate noise / register / initialize maps ---
    data, maps = preproc(data, opt)

    # --- initialize weights (RLS) ---
    mean_shape = maps.decay.volume.shape
    rls = None
    sumrls = 0
    if opt.regularization.norm.lower() in ('tv', 'jtv'):
        rls_shape = mean_shape
        if opt.regularization.norm.lower() == 'tv':
            rls_shape = (len(maps),) + rls_shape
        rls = ParameterMap(rls_shape, fill=1, **backend).volume
        sumrls = rls.sum(dtype=torch.double)

    # --- prepare regularization factor ---
    lam = opt.regularization.factor
    lam = core.utils.make_list(lam)
    if len(lam) > 1:
        *lam, lam_decay = lam
    else:
        lam_decay = lam[0]
    lam = core.utils.make_list(lam, len(maps)-1)
    lam.append(lam_decay)

    # --- compute derivatives ---
    grad = torch.empty((len(data) + 1,) + mean_shape, **backend)
    hess = torch.empty((len(data)*2 + 1,) + mean_shape, **backend)

    if opt.regularization.norm not in ('tv', 'jtv'):
        # no reweighting -> do more gauss-newton updates instead
        opt.optim.max_iter_gn *= opt.optim.max_iter_rls
        opt.optim.max_iter_rls = 1

    print('{:^3s} | {:^3s} | {:^12s} + {:^12s} + {:^12s} = {:^12s} | {:^2s}'
          .format('rls', 'gn', 'fit', 'reg', 'rls', 'crit', 'ls'))

    for n_iter_rls in range(opt.optim.max_iter_rls):

        multi_rls = rls if opt.regularization.norm == 'tv' \
                    else [rls] * len(maps)

        # --- Gauss Newton loop ---
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
            reg = 0
            for i, (map, weight, l) in enumerate(zip(maps, multi_rls, lam)):
                reg1, g1 = _nonlin_reg(map, weight, l)
                reg += reg1
                grad[i] += g1

            # --- load diagonal of the Hessian ---
            hess = hessian_loaddiag(hess)

            # --- gauss-newton ---
            deltas = _nonlin_solve(hess, grad, multi_rls, lam, maps.affine, opt)

            # --- line search ---
            if not opt.optim.max_iter_ls:
                n_iter_ls = 0
                for map, delta in zip(maps, deltas):
                    map.volume -= delta
                    if map.min is not None or map.max is not None:
                        map.volume.clamp_(map.min, map.max)
            else:
                armijo = 1.
                ok = False
                crit0 = crit
                reg0 = reg
                maps0 = maps
                for n_iter_ls in range(opt.optim.max_iter_ls):
                    maps = maps0.deepcopy()
                    for map, delta in zip(maps, deltas):
                        map.volume -= armijo * delta
                        if map.min is not None or map.max is not None:
                            map.volume.clamp_(map.min, map.max)
                    crit = sum(_nonlin_gradient(contrast, intercept, maps.decay, opt, do_grad=False)
                               for contrast, intercept in zip(data, maps.intercepts))
                    reg = sum(_nonlin_reg(map, weight, l, do_grad=False)
                              for map, weight, l in zip(maps, multi_rls, lam))
                    if crit + reg > crit0 + reg0:
                        ok = True
                        break
                    else:
                        armijo /= 2.
                if not ok:
                    reg = reg0
                    crit = crit0
                    maps = maps0
                    break

            print('{:3d} | {:3d} | {:12.6g} + {:12.6g} + {:12.6g} = {:12.6g} | {:2d}'
                  .format(n_iter_rls, n_iter_gn, crit, reg, sumrls,
                          crit + reg + sumrls, n_iter_ls))

        # --- Update RLS weights ---
        if opt.regularization.norm in ('tv', 'jtv'):
            rls = _nonlin_rls(maps, lam, opt.regularization.norm)
            sumrls = -0.5 * rls.sum(dtype=torch.double)
            rls = rls.reciprocal_()

    # --- Prepare output ---
    return postproc(maps, data)


def grad_obs(obs, fit, prec=1., do_grad=True):
    """Compute the gradient of the data term

    Parameters
    ----------
    obs : (P, ...) tensor_like
        Observed
    fit : (P, ...) tensor_like
        Denoised
    prec : () or (P,) or tensor_like
        Precision of the noise
    do_grad : bool, default=True
        Compute gradient and Hessian

    Returns
    -------
    crit : log-likelihood
    grad : (P, ...) tensor, optional
    hess : (P, ...) or (P, P) tensor, optional

    """
    obs = torch.as_tensor(obs)
    backend = dict(dtype=obs.dtype, device=obs.device)
    fit = torch.as_tensor(fit, **backend)
    prec = torch.as_tensor(prec, **backend)

    res = fit - obs
    msk = torch.isfinite(fit) & torch.isfinite(obs)
    res[~msk] = 0
    mres = last_to_first(first_to_last(res) * prec)
    res *= mres

    crit = -0.5 * res.sum(dtype=torch.double)
    del res

    if do_grad:
        grad = mres
        hess = prec
        return crit, grad, hess
    else:
        return crit


def grad_reg(fit, rls=None, lam=1., vx=1., do_grad=True):
    """Compute the gradient of the regularisation term.

    The regularisation term has the form:
    `-0.5 * lam * sum(w[i] * (g+[i]**2 + g-[i]**2) / 2)`
    where `i` indexes a voxel, `lam` is the regularisation factor,
    `w[i]` is the RLS weight, `g+` and `g-` are the forward and
    backward spatial gradients of the parameter map.

    Parameters
    ----------
    fit : (*shape) ParameterMap
        Parameter map
    rls : (*shape) tensor, optional
        Weights from the reweighted least squares schme
    lam : float, default=1
        Regularisation factor
    vx : float or sequence[float], defqult=1.
        Voxel size
    do_grad : bool, default=True
        Return both the criterion and gradient

    Returns
    -------
    reg : () tensor
        Regularisation term
    grad : (*shape) tensor, if do_grqd
        Gradient with respect to the parameter map

    """

    grad_fwd = spatial.diff(fit, dim=[0, 1, 2], voxel_size=vx, side='f')
    grad_bwd = spatial.diff(fit, dim=[0, 1, 2], voxel_size=vx, side='b')
    if rls is not None:
        grad_fwd *= rls[..., None]
        grad_bwd *= rls[..., None]
    grad_fwd = spatial.div(grad_fwd, dim=[0, 1, 2], voxel_size=vx, side='f')
    grad_bwd = spatial.div(grad_bwd, dim=[0, 1, 2], voxel_size=vx, side='b')

    grad = grad_fwd
    grad += grad_bwd
    grad *= lam / (2*3)   # average across directions (3) and side (2)

    if do_grad:
        reg = (fit * grad).sum(dtype=torch.double)
        return -0.5 * reg, grad
    else:
        grad *= fit
        return -0.5 * grad.sum(dtype=torch.double)


def solve(hess, grad, rls, lam, vx, opt):
    """Solve the regularized linear system

    Parameters
    ----------
    hess : (P|P*P|P*(P+1)//2, *shape) tensor_like
        Hessian matrix: can be diagonal or symmetric (sparse or full)
    grad : (P, *shape) tensor_like
        Gradient
    rls : ([P], *shape) tensor_like
        L1 weights: either one per parameter (TV) or one shared
        across parameters (JTV)
    lam : float or (P,) sequence[float]
        Regularisation
    vx : float or  (D,) sequence[float]
        Voxel size
    opt : Options
        Solver options.

    Returns
    -------
    delta : (P+1, *shape) tensor

    """
    grad = torch.as_tensor(grad)
    hess = torch.as_tensor(hess)
    rls = torch.as_tensor(rls) if rls is not None else None
    nb_param = grad.shape[0]
    nb_dim = grad.dim() - 1
    lam = core.pyutils.make_list(lam, nb_param)
    vx = core.pyutils.make_list(vx, nb_dim)

    def hess_fn(x):
        result = hessian_matvec(hess, x)
        for i, (map, weight, l, v) in enumerate(zip(x, rls, lam, vx)):
            _, res1 = grad_reg(map, weight, l, v)
            result[i] += res1
        return result

    result = core.optim.cg(hess_fn, grad,
                           verbose=True, stop='norm',
                           max_iter=opt.max_iter_cg,
                           tolerance=opt.tolerance_cg)
    return result
